# Project instructions

## Polypack memory

Use the connected Polypack MCP when working on this repository:

- Before tasks involving prior decisions, project history, preferences, or accumulated context, call `mcp__polypack__memory_recall` with context `polypack-mcp`.
- Store durable architectural decisions, important discoveries, workflows, and user preferences with `mcp__polypack__memory_store` using context `polypack-mcp`.
- After completing a meaningful task, store a concise summary of the outcome when it is likely to help future work.
- Use `mcp__polypack__memory_context` when a bounded working-memory set is more useful than a keyword search.
- Do not store secrets, credentials, personal sensitive data, transient debugging output, or trivial conversation.
- If Polypack is unavailable or a call fails, continue the task and mention the limitation only when it affects the result.

## Tool workflow

- Use `memory_recall` for targeted searches and `memory_context` for bounded working context.
- Use `memory_store` for durable facts, decisions, preferences, and outcomes; use `procedural` for conventions/preferences, `semantic` for stable facts, `episodic` for task outcomes, and `entity` for named objects.
- Use `memory_get` for exact ID lookup and `memory_update` for context, confidence, provenance, or metadata. Use `memory_supersede` when content changes so history is retained.
- Use `memory_link` with `RESPONDS_TO` for replies, handoffs, reviews, and fixes; use `memory_thread` or bounded neighbor recall to follow a chain. Use `memory_unlink` to correct an edge.
- Use `memory_list_contexts` to discover namespaces. Use `memory_feedback` after retrieved memories help or mislead.
- Treat `memory_delete` as permanent: call it only deliberately with `confirm=true`; prefer `memory_suppress` when history should remain available.
