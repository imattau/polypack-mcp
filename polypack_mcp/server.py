"""FastMCP server for Polypack adaptive memory."""

import argparse
import base64
import hashlib
import json
import os
import secrets
import signal
import time
import uuid
from contextlib import contextmanager
from threading import Condition, Event, RLock, Thread
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .backend import MemoryBackend, MemoryClass


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


class ReadWriteLock:
    """Allow concurrent reads while keeping graph mutations exclusive."""

    def __init__(self) -> None:
        self._condition = Condition(RLock())
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def read(self):
        with self._condition:
            while self._writer or self._waiting_writers:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @contextmanager
    def write(self):
        with self._condition:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers:
                    self._condition.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1
        try:
            yield
        finally:
            with self._condition:
                self._writer = False
                self._condition.notify_all()


def _storage_limit_bytes() -> int:
    raw = os.environ.get("POLYPACK_MCP_STORAGE_LIMIT_BYTES", "0").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        print("polypack: ignoring invalid POLYPACK_MCP_STORAGE_LIMIT_BYTES", flush=True)
        return 0
    return max(0, value)


def _start_storage_janitor(backend: MemoryBackend, operation_lock: ReadWriteLock):
    limit_bytes = _storage_limit_bytes()
    cleanup = getattr(backend, "cleanup_storage", None)
    if limit_bytes <= 0 or cleanup is None:
        return None, None
    stop = Event()
    interval = max(30, int(os.environ.get("POLYPACK_MCP_CLEANUP_INTERVAL_SECONDS", "300")))

    def run() -> None:
        while not stop.is_set():
            try:
                with operation_lock.write():
                    result = cleanup(limit_bytes)
                if result.get("removed", 0):
                    print(
                        f"polypack: storage cleanup removed {result['removed']} memories "
                        f"({result['beforeBytes']} -> {result['afterBytes']} bytes)",
                        flush=True,
                    )
            except Exception as exc:  # cleanup must never take down the MCP server
                print(f"polypack: storage cleanup failed: {type(exc).__name__}: {exc}", flush=True)
            if stop.wait(interval):
                return

    thread = Thread(target=run, name="polypack-storage-janitor", daemon=True)
    thread.start()
    return stop, thread
from .backend import InMemoryBackend, MemoryBackend
from .service import MemoryService
from .setup import install_user_service, print_client_config

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_MAX_OAUTH_CLIENTS = 256
_AUTH_CODE_TTL_SECONDS = 300
_ACCESS_TOKEN_TTL_SECONDS = 86400
_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type, mcp-protocol-version",
}
_NO_STORE_HEADERS = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
}


class _NormalizeMCPPathMiddleware:
    """Rewrite /mcp to /mcp/ so clients avoid a 307 redirect from Starlette Mount."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") == "http" and scope.get("path") == "/mcp":
            scope = dict(scope, path="/mcp/", raw_path=b"/mcp/")
        await self.app(scope, receive, send)


def _is_loopback_redirect_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()
    if not scheme:
        return False
    if scheme in {"http", "https"}:
        return (parsed.hostname or "").lower() in _LOOPBACK_HOSTS
    return True


def _redirect_uri_matches(requested: str, registered_list: list[str]) -> bool:
    if requested in registered_list:
        return True
    req = urlparse(requested)
    if req.scheme.lower() != "http" or (req.hostname or "").lower() not in _LOOPBACK_HOSTS:
        return False
    for entry in registered_list:
        reg = urlparse(entry)
        if (
            reg.scheme.lower() == "http"
            and (reg.hostname or "").lower() in _LOOPBACK_HOSTS
            and req.path == reg.path
        ):
            return True
    return False


def _append_query_params(url: str, extra: dict[str, str]) -> str:
    parsed = urlparse(url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    for key, value in extra.items():
        if value is not None:
            query.append((key, value))
    return urlunparse(parsed._replace(query=urlencode(query)))


def _parse_basic_auth(header_value: str | None) -> tuple[str | None, str | None]:
    if not header_value or not header_value.lower().startswith("basic "):
        return None, None
    encoded = header_value[6:].strip()
    try:
        decoded = base64.b64decode(encoded).decode("utf-8")
    except Exception:
        return None, None
    if ":" not in decoded:
        return None, None
    client_id, client_secret = decoded.split(":", 1)
    return client_id, client_secret


def _register_loopback_oauth_routes(mcp, host: str, port: int) -> None:
    """Expose loopback OAuth 2.1 discovery, DCR, and PKCE endpoints without gating /mcp."""
    from starlette.responses import JSONResponse, RedirectResponse, Response
    from starlette.routing import Mount

    lock = RLock()
    clients: dict[str, dict] = {}
    auth_codes: dict[str, dict] = {}
    refresh_tokens: dict[str, dict] = {}

    def _base_url(request) -> str:
        if request.url.netloc:
            return f"{request.url.scheme}://{request.url.netloc}".rstrip("/")
        return f"http://{host}:{port}"

    def _json_response(payload: dict, status_code: int = 200, *, no_store: bool = False) -> JSONResponse:
        headers = dict(_CORS_HEADERS)
        if no_store:
            headers.update(_NO_STORE_HEADERS)
        return JSONResponse(payload, status_code=status_code, headers=headers)

    def _options_response() -> Response:
        return Response(status_code=204, headers=dict(_CORS_HEADERS))

    def _oauth_error(error: str, description: str, status_code: int = 400) -> JSONResponse:
        return _json_response(
            {"error": error, "error_description": description},
            status_code=status_code,
            no_store=True,
        )

    async def auth_server_metadata(request):
        if request.method == "OPTIONS":
            return _options_response()
        base = _base_url(request)
        return _json_response(
            {
                "issuer": base,
                "authorization_endpoint": f"{base}/authorize",
                "token_endpoint": f"{base}/token",
                "registration_endpoint": f"{base}/register",
                "revocation_endpoint": f"{base}/revoke",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "token_endpoint_auth_methods_supported": ["none", "client_secret_post", "client_secret_basic"],
                "code_challenge_methods_supported": ["S256", "plain"],
            }
        )

    async def protected_resource_metadata(request):
        if request.method == "OPTIONS":
            return _options_response()
        base = _base_url(request)
        return _json_response(
            {
                "resource": f"{base}/mcp/",
                "authorization_servers": [base],
                "bearer_methods_supported": ["header"],
            }
        )

    async def register_client(request):
        if request.method == "OPTIONS":
            return _options_response()
        raw_body = await request.body()
        if not raw_body.strip():
            body = {}
        else:
            try:
                body = json.loads(raw_body)
            except ValueError:
                return _oauth_error("invalid_client_metadata", "Request body must be a JSON object")
            if not isinstance(body, dict):
                return _oauth_error("invalid_client_metadata", "Request body must be a JSON object")

        if "redirect_uris" in body:
            redirect_uris = body["redirect_uris"]
            if (
                not isinstance(redirect_uris, list)
                or not redirect_uris
                or not all(isinstance(uri, str) and uri.strip() for uri in redirect_uris)
            ):
                return _oauth_error("invalid_redirect_uri", "redirect_uris must be a non-empty list of URI strings")
        else:
            redirect_uris = []

        raw_client_id = body.get("client_id")
        client_id = raw_client_id if isinstance(raw_client_id, str) and raw_client_id.strip() else str(uuid.uuid4())
        auth_method = body.get("token_endpoint_auth_method") or "none"
        grant_types = body.get("grant_types") or ["authorization_code", "refresh_token"]
        response_types = body.get("response_types") or ["code"]

        record = dict(body)
        record.update(
            {
                "client_id": client_id,
                "client_id_issued_at": int(time.time()),
                "redirect_uris": redirect_uris,
                "token_endpoint_auth_method": auth_method,
                "grant_types": grant_types,
                "response_types": response_types,
            }
        )
        if auth_method != "none":
            record["client_secret"] = secrets.token_hex(32)
            record["client_secret_expires_at"] = 0

        with lock:
            clients.pop(client_id, None)
            while len(clients) >= _MAX_OAUTH_CLIENTS:
                oldest = next(iter(clients))
                clients.pop(oldest, None)
            clients[client_id] = record

        return _json_response(record, status_code=201, no_store=True)

    async def authorize(request):
        params = dict(request.query_params)
        if request.method == "POST":
            form = await request.form()
            for key, value in form.items():
                if isinstance(value, str):
                    params[key] = value

        response_type = params.get("response_type", "code")
        if response_type != "code":
            return _oauth_error("unsupported_response_type", "Only response_type=code is supported")

        client_id = params.get("client_id")
        if not client_id:
            return _oauth_error("invalid_request", "client_id is required")

        requested_redirect_uri = params.get("redirect_uri")
        redirect_uri_provided = bool(requested_redirect_uri)

        with lock:
            client = clients.get(client_id)

        if client is not None:
            registered_uris = client.get("redirect_uris") or []
            if requested_redirect_uri:
                if registered_uris and not _redirect_uri_matches(requested_redirect_uri, registered_uris):
                    return _oauth_error("invalid_request", "redirect_uri does not match registered client URIs")
                if not registered_uris and not _is_loopback_redirect_uri(requested_redirect_uri):
                    return _oauth_error("invalid_request", "redirect_uri must be a loopback or native URI")
                redirect_uri = requested_redirect_uri
            elif len(registered_uris) == 1:
                redirect_uri = registered_uris[0]
            else:
                return _oauth_error("invalid_request", "redirect_uri is required")
        else:
            if not requested_redirect_uri or not _is_loopback_redirect_uri(requested_redirect_uri):
                return _oauth_error("invalid_request", "redirect_uri must be a loopback or native URI")
            redirect_uri = requested_redirect_uri

        code = secrets.token_urlsafe(32)
        now = time.time()
        with lock:
            expired = [k for k, v in auth_codes.items() if v["expires_at"] < now]
            for k in expired:
                auth_codes.pop(k, None)
            auth_codes[code] = {
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "redirect_uri_provided": redirect_uri_provided,
                "code_challenge": params.get("code_challenge"),
                "code_challenge_method": params.get("code_challenge_method") or "S256",
                "scope": params.get("scope"),
                "expires_at": now + _AUTH_CODE_TTL_SECONDS,
            }

        extra = {"code": code}
        if params.get("state") is not None:
            extra["state"] = params["state"]
        redirect_url = _append_query_params(redirect_uri, extra)
        return RedirectResponse(url=redirect_url, status_code=302, headers=dict(_NO_STORE_HEADERS))

    async def exchange_token(request):
        if request.method == "OPTIONS":
            return _options_response()

        params: dict[str, str] = {k: v for k, v in request.query_params.items()}
        content_type = (request.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            try:
                body = await request.json()
            except ValueError:
                body = {}
            if isinstance(body, dict):
                for key, value in body.items():
                    if isinstance(value, str):
                        params[key] = value
        else:
            try:
                form = await request.form()
                for key, value in form.items():
                    if isinstance(value, str):
                        params[key] = value
            except Exception:
                pass

        basic_id, basic_secret = _parse_basic_auth(request.headers.get("authorization"))
        req_client_id = params.get("client_id") or basic_id
        req_client_secret = params.get("client_secret") if "client_secret" in params else basic_secret

        grant_type = params.get("grant_type")
        if grant_type == "authorization_code":
            code = params.get("code")
            with lock:
                auth_code = auth_codes.pop(code, None) if code else None
            if not auth_code or auth_code["expires_at"] < time.time():
                return _oauth_error("invalid_grant", "Invalid or expired authorization code")

            if req_client_id and req_client_id != auth_code["client_id"]:
                return _oauth_error("invalid_grant", "client_id does not match authorization code")

            with lock:
                client = clients.get(auth_code["client_id"])
            if client and client.get("client_secret") and req_client_secret is not None:
                if not secrets.compare_digest(req_client_secret, client["client_secret"]):
                    return _oauth_error("unauthorized_client", "Invalid client_secret")

            req_redirect_uri = params.get("redirect_uri")
            if auth_code["redirect_uri_provided"] and req_redirect_uri and req_redirect_uri != auth_code["redirect_uri"]:
                return _oauth_error("invalid_request", "redirect_uri did not match authorization request")

            code_challenge = auth_code.get("code_challenge")
            if code_challenge:
                code_verifier = params.get("code_verifier")
                if not code_verifier:
                    return _oauth_error("invalid_grant", "code_verifier is required")
                method = auth_code.get("code_challenge_method", "S256")
                if method == "S256":
                    try:
                        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
                    except UnicodeEncodeError:
                        return _oauth_error("invalid_grant", "incorrect code_verifier")
                    computed = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
                elif method == "plain":
                    computed = code_verifier
                else:
                    return _oauth_error("invalid_grant", f"Unsupported code_challenge_method: {method}")
                if not secrets.compare_digest(computed, code_challenge):
                    return _oauth_error("invalid_grant", "incorrect code_verifier")

            access_token = secrets.token_urlsafe(32)
            refresh_token = secrets.token_urlsafe(32)
            with lock:
                refresh_tokens[refresh_token] = {
                    "client_id": auth_code["client_id"],
                    "scope": auth_code.get("scope"),
                }
            payload = {
                "access_token": access_token,
                "token_type": "Bearer",
                "expires_in": _ACCESS_TOKEN_TTL_SECONDS,
                "refresh_token": refresh_token,
            }
            if auth_code.get("scope"):
                payload["scope"] = auth_code["scope"]
            return _json_response(payload, status_code=200, no_store=True)

        if grant_type == "refresh_token":
            old_refresh = params.get("refresh_token")
            if not old_refresh:
                return _oauth_error("invalid_request", "refresh_token is required")
            with lock:
                prior = refresh_tokens.pop(old_refresh, None)
                new_access = secrets.token_urlsafe(32)
                new_refresh = secrets.token_urlsafe(32)
                scope = params.get("scope") or (prior.get("scope") if prior else None)
                refresh_tokens[new_refresh] = {
                    "client_id": req_client_id or (prior.get("client_id") if prior else None),
                    "scope": scope,
                }
            payload = {
                "access_token": new_access,
                "token_type": "Bearer",
                "expires_in": _ACCESS_TOKEN_TTL_SECONDS,
                "refresh_token": new_refresh,
            }
            if scope:
                payload["scope"] = scope
            return _json_response(payload, status_code=200, no_store=True)

        return _oauth_error("unsupported_grant_type", f"Unsupported grant_type: {grant_type}")

    async def revoke_token(request):
        if request.method == "OPTIONS":
            return _options_response()
        return _json_response({}, status_code=200)

    for path in (
        "/.well-known/oauth-authorization-server",
        "/.well-known/oauth-authorization-server/mcp",
        "/mcp/.well-known/oauth-authorization-server",
    ):
        mcp.custom_route(path, methods=["GET", "OPTIONS"], include_in_schema=False)(auth_server_metadata)

    for path in (
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-protected-resource/mcp",
        "/mcp/.well-known/oauth-protected-resource",
    ):
        mcp.custom_route(path, methods=["GET", "OPTIONS"], include_in_schema=False)(protected_resource_metadata)

    for path in ("/register", "/mcp/register"):
        mcp.custom_route(path, methods=["POST", "OPTIONS"], include_in_schema=False)(register_client)

    for path in ("/authorize", "/mcp/authorize"):
        mcp.custom_route(path, methods=["GET", "POST"], include_in_schema=False)(authorize)

    for path in ("/token", "/mcp/token"):
        mcp.custom_route(path, methods=["POST", "OPTIONS"], include_in_schema=False)(exchange_token)

    for path in ("/revoke", "/mcp/revoke"):
        mcp.custom_route(path, methods=["POST", "OPTIONS"], include_in_schema=False)(revoke_token)

    orig_streamable_http_app = mcp.streamable_http_app

    def streamable_http_app_with_oauth():
        starlette_app = orig_streamable_http_app()
        starlette_app.router.routes[:] = [
            *[route for route in starlette_app.router.routes if not isinstance(route, Mount)],
            *[route for route in starlette_app.router.routes if isinstance(route, Mount)],
        ]
        starlette_app.add_middleware(_NormalizeMCPPathMiddleware)
        return starlette_app

    mcp.streamable_http_app = streamable_http_app_with_oauth

def create_server(backend: MemoryBackend | None = None, host: str = "127.0.0.1", port: int = 8765):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -e '.[polypack]'") from exc
    backend = backend or InMemoryBackend()
    service = MemoryService(backend)
    # FastMCP may execute synchronous tools in worker threads. Polypack's
    # graph transaction boundary is process-local: reads may overlap, but
    # mutations must exclude both reads and other mutations.
    operation_lock = ReadWriteLock()
    # This server has no per-client server-side state or server-initiated
    # requests. Stateless HTTP avoids stale Mcp-Session-Id failures when
    # multiple clients connect, reconnect, or join an existing endpoint.
    mcp = FastMCP("Polypack", host=host, port=port, stateless_http=True)

    @mcp.tool()
    def memory_store(content: str, memory_class: MemoryClass = "semantic", context: str | None = None,
                     confidence: float = 1.0, provenance: dict | None = None, metadata: dict | None = None) -> dict:
        """Store durable project memory.

        Use procedural for preferences, conventions, and decisions; semantic for
        stable facts; episodic for events or task outcomes; and entity for named
        people, projects, or objects. Use a stable context for project memory.
        """
        with operation_lock.write():
            return service.store(content, memory_class, context, confidence, provenance, metadata)

    @mcp.tool()
    def memory_get(memory_id: str) -> dict:
        """Fetch one memory by exact ID, including its current state and revision."""
        with operation_lock.read():
            return backend.get(memory_id)

    @mcp.tool()
    def memory_update(memory_id: str, patch: dict, expected_revision: int | None = None) -> dict:
        """Update mutable memory fields without changing its content.

        Supported fields are context, confidence, provenance, and metadata. Use
        memory_supersede when the memory content itself needs to change.
        expected_revision prevents overwriting a concurrent update.
        """
        if not isinstance(patch, dict) or not patch:
            raise ValueError("patch must be a non-empty object")
        with operation_lock.write():
            return backend.update(memory_id, patch, expected_revision=expected_revision)

    @mcp.tool()
    def memory_list_contexts() -> dict:
        """List context namespaces and the number of memories in each."""
        with operation_lock.read():
            return backend.list_contexts()

    @mcp.tool()
    def memory_delete(memory_id: str, confirm: bool = False,
                      expected_revision: int | None = None) -> dict:
        """Permanently delete one memory and its graph edges.

        This destructive operation requires confirm=true. Prefer
        memory_suppress when retaining history is useful.
        """
        if not confirm:
            raise ValueError("memory_delete requires confirm=true")
        with operation_lock.write():
            return backend.delete(memory_id, expected_revision=expected_revision)

    @mcp.tool()
    def memory_recall(query: str, context: str | None = None, limit: int = 10,
                      strict_context: bool = False, include_neighbors: bool = False,
                      edge_types: list[str] | None = None, depth: int = 1,
                      neighbor_limit: int = 1,
                      token_budget: int | None = None) -> dict:
        """Search memories by text, context, activation, and confidence.

        Use this for a targeted question. Use memory_context to assemble working
        context. Context is soft by default; use strict_context=true for isolation.
        With include_neighbors=true, bounded graph neighbors are hydrated into the
        result. Filter relationships with edge_types such as RESPONDS_TO. The
        neighbor_limit bounds hydrated neighbors; limit remains the total result
        count. depth is silently clamped to at most 3 hops regardless of the value
        passed; metadata reports the depth actually applied. Metadata also reports
        when additional neighbors were available.
        """
        if token_budget is not None and token_budget <= 0:
            raise ValueError("token_budget must be greater than zero")
        if neighbor_limit < 0:
            raise ValueError("neighbor_limit must be zero or greater")
        applied_depth = max(0, min(depth, 3))
        with operation_lock.read():
            result = backend.recall(query, context=context, strict_context=strict_context,
                                    limit=max(1, min(limit, 100)), include_neighbors=include_neighbors,
                                    edge_types=edge_types, depth=applied_depth,
                                    neighbor_limit=min(neighbor_limit, 100),
                                    token_budget=token_budget)
        if include_neighbors:
            result["metadata"]["depthRequested"] = depth
            result["metadata"]["depthApplied"] = applied_depth
        return result

    @mcp.tool()
    def memory_thread(start_id: str, max_depth: int = 10) -> dict:
        """Walk the RESPONDS_TO chain from a starting memory ID and return the thread.

        Retrieves connected memories in both directions along RESPONDS_TO edges,
        returning them sorted chronologically.
        """
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero")
        with operation_lock.read():
            return backend.memory_thread(start_id, max_depth=min(max_depth, 100))

    @mcp.tool()
    def memory_context(context: str, limit: int = 10, token_budget: int | None = None,
                       strict_context: bool = False) -> dict:
        """Return a working-memory set selected by activation and estimated-token budget.

        token_budget is an estimated-token budget. Each returned memory fits wholly
        within the remaining budget; budgets must be greater than zero.
        """
        if token_budget is not None and token_budget <= 0:
            raise ValueError("token_budget must be greater than zero")
        with operation_lock.read():
            return backend.context(context, limit=max(1, min(limit, 100)), token_budget=token_budget,
                                   strict_context=strict_context)

    @mcp.tool()
    def memory_feedback(memory_id: str, useful: bool, agent_id: str = "default") -> dict:
        """Record whether a retrieved memory helped this task.

        Call this after using a recalled memory when it was useful or misleading.
        """
        with operation_lock.write():
            return backend.feedback(memory_id, useful, agent_id=agent_id)

    @mcp.tool()
    def memory_suppress(memory_id: str, amount: float = 0.5) -> dict:
        """Inhibit a stale or unhelpful memory without deleting it."""
        with operation_lock.write():
            return backend.suppress(memory_id, amount=amount)

    @mcp.tool()
    def memory_supersede(new_memory_id: str, old_memory_id: str) -> dict:
        """Replace an outdated fact while retaining its history."""
        with operation_lock.write():
            return backend.supersede(new_memory_id, old_memory_id)

    @mcp.tool()
    def memory_consolidate(source_ids: list[str], content: str, context: str | None = None,
                           memory_class: MemoryClass = "semantic", confidence: float = 1.0) -> dict:
        """Consolidate source memories into one durable higher-level memory."""
        with operation_lock.write():
            return backend.consolidate(source_ids, content, context=context, memory_class=memory_class, confidence=confidence)

    @mcp.tool()
    def memory_link(source_memory_id: str, target_memory_id: str,
                    relationship: str = "RESPONDS_TO") -> dict:
        """Connect two memories with an explicit graph relationship.

        Use RESPONDS_TO when a new handoff, verification, or fix addresses an
        earlier memory. Use memory_recall(include_neighbors=true) to retrieve
        linked memories with relationship metadata.
        """
        with operation_lock.write():
            return backend.link(source_memory_id, target_memory_id, relationship)

    @mcp.tool()
    def memory_store_with_link(content: str, target_memory_id: str, relationship: str = "RESPONDS_TO",
                               memory_class: MemoryClass = "semantic", context: str | None = None,
                               confidence: float = 1.0, provenance: dict | None = None,
                               metadata: dict | None = None, reverse: bool = False) -> dict:
        """Store a new memory and link it to an existing memory in one call.

        Combines memory_store and memory_link for the common case of storing a
        reply, verification, or fix that responds to an earlier memory. By
        default the new memory is the link source (new -relationship-> target);
        set reverse=true to make the new memory the link target instead.
        """
        with operation_lock.write():
            backend.get(target_memory_id)
            stored = service.store(content, memory_class, context, confidence, provenance, metadata)
            source_id, target_id = (target_memory_id, stored["id"]) if reverse else (stored["id"], target_memory_id)
            link = backend.link(source_id, target_id, relationship)
            return {"memory": stored, "link": link}

    @mcp.tool()
    def memory_unlink(source_memory_id: str, target_memory_id: str,
                      relationship: str | None = None) -> dict:
        """Remove graph edges between two memories.

        If relationship is omitted, all directed edges from source to target
        are removed.
        """
        with operation_lock.write():
            return backend.unlink(source_memory_id, target_memory_id, relationship)

    @mcp.tool()
    def memory_store_batch(memories: list[dict]) -> list[dict]:
        """Store multiple durable project memories in a single batch.

        Each item in the list should be a dict with 'content' and optional
        'memory_class', 'context', 'confidence', 'provenance', and 'metadata' keys.
        """
        with operation_lock.write():
            return service.store_batch(memories)

    @mcp.tool()
    def memory_link_batch(links: list[dict]) -> list[dict]:
        """Connect multiple pairs of memories with explicit graph relationships in a single batch.

        Each item in the list should be a dict with 'source_memory_id', 'target_memory_id',
        and optional 'relationship' (defaults to 'RESPONDS_TO') keys.
        """
        with operation_lock.write():
            return backend.link_batch(links)

    @mcp.tool()
    def graph_query(operation: str, id: str | None = None, source: str | None = None,
                    target: str | None = None, type: str | None = None) -> dict:
        """Inspect graph neighbors/schema or perform an advanced edge operation.

        Prefer memory_link for normal memory relationships. Supported operations
        are neighbors, add_edge, schema, and relationship_diagnostics. RESPONDS_TO
        graph edges are authoritative; diagnostics identifies legacy
        provenance-only relationships.
        """
        args = {k: v for k, v in {"id": id, "source": source, "target": target, "type": type}.items() if v is not None}
        lock = operation_lock.write if operation == "add_edge" else operation_lock.read
        with lock():
            return backend.graph_query(operation, **args)

    @mcp.resource("polypack://memory/context/{context}")
    def context_resource(context: str) -> str:
        with operation_lock.read():
            return json.dumps(backend.context(context), indent=2)

    @mcp.resource("polypack://memory/active")
    def active_resource() -> str:
        with operation_lock.read():
            return json.dumps(backend.context("", limit=20), indent=2)

    @mcp.resource("polypack://graph/schema")
    def schema_resource() -> str:
        with operation_lock.read():
            return json.dumps(backend.graph_query("schema"), indent=2)

    @mcp.resource("polypack://stats")
    def stats_resource() -> str:
        with operation_lock.read():
            return json.dumps(backend.stats(), indent=2)

    @mcp.resource("polypack://help/workflow")
    def workflow_help_resource() -> str:
        return """Polypack MCP agent workflow

1. Use memory_recall for a targeted question.
2. Use memory_context when assembling working context for a task.
3. Store durable discoveries, decisions, preferences, and outcomes with memory_store.
4. Use procedural for conventions/preferences, semantic for facts, episodic for
   events/outcomes, and entity for named objects.
5. Use memory_get for exact IDs and memory_update for context, confidence,
   provenance, or metadata. Use memory_supersede when content changes.
6. Link replies, verifications, and fixes with memory_link using RESPONDS_TO;
   correct mistakes with memory_unlink. Use memory_store_with_link to store
   and link in one call.
7. Follow a handoff chain with memory_recall(include_neighbors=true,
   edge_types=[RESPONDS_TO], depth=1 or 2, neighbor_limit=1 or more).
8. Call memory_feedback after a retrieved memory materially helps or misleads.
9. Use memory_list_contexts to discover namespaces. Use memory_delete only for
   deliberate permanent cleanup with confirm=true.

Graph edges are authoritative for relationships. Use graph_query with
operation=relationship_diagnostics to audit legacy provenance.responds_to fields.

Context is a soft filter unless strict_context=true. If recall is empty, retry
with a broader query and strict_context=false before assuming the store is empty.
"""
    _register_loopback_oauth_routes(mcp, host=host, port=port)
    janitor_stop, janitor_thread = _start_storage_janitor(backend, operation_lock)
    mcp._polypack_janitor_stop = janitor_stop
    mcp._polypack_janitor_thread = janitor_thread
    return mcp

def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "embeddings":
        from .embedding_cli import main as embedding_main
        raise SystemExit(embedding_main(sys.argv[2:]))
    parser = argparse.ArgumentParser(description="Polypack adaptive-memory MCP server")
    parser.add_argument("command", nargs="?", choices=("serve", "setup"), default="serve")
    parser.add_argument("--transport", choices=("stdio", "sse", "streamable-http"), default="stdio")
    parser.add_argument("--store", help="Polypack directory for durable storage")
    parser.add_argument("--host", default="127.0.0.1", help="Host used by the HTTP server")
    parser.add_argument("--port", type=int, default=8765, help="Port used by the shared HTTP endpoint")
    parser.add_argument("--storage-limit-mb", type=int, default=0,
                        help="Clean low-value memories when the graph exceeds this many MiB")
    parser.add_argument("--no-start", action="store_true", help="Write the user service without starting it")
    args = parser.parse_args()
    if args.command == "setup":
        store = args.store or "~/.local/share/polypack-mcp"
        unit = install_user_service(store, port=args.port, start=not args.no_start,
                                    storage_limit_mb=args.storage_limit_mb)
        print(f"Installed {unit}")
        print_client_config(args.port)
        return
    backend = None
    if args.store:
        from .backend import PolypackBackend
        from .embeddings import HTTPEmbeddingProvider
        from polypack import PolyGraph
        embedding_url = os.environ.get("POLYPACK_MCP_EMBEDDING_URL")
        provider = HTTPEmbeddingProvider(embedding_url) if embedding_url else None
        backend = PolypackBackend(PolyGraph.open(args.store), embedding_provider=provider)
    server = create_server(backend, host=args.host, port=args.port)
    try:
        if backend is not None:
            # Convert SIGTERM into normal Python shutdown so the finally block
            # can flush mutations before the process exits.
            signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
        server.run(transport=args.transport)
    finally:
        janitor_stop = getattr(server, "_polypack_janitor_stop", None)
        janitor_thread = getattr(server, "_polypack_janitor_thread", None)
        if janitor_stop is not None:
            janitor_stop.set()
        if janitor_thread is not None:
            janitor_thread.join(timeout=5)
        if backend is not None:
            backend.close_store()


if __name__ == "__main__":
    main()
