#!/usr/bin/env bash
set -euo pipefail

version="${1:?usage: build.sh VERSION OUTPUT_DIR}"
output_dir="${2:?usage: build.sh VERSION OUTPUT_DIR}"
root="$(mktemp -d)"
trap 'rm -rf "$root"' EXIT

command -v rpmbuild >/dev/null || { echo "rpmbuild is required" >&2; exit 1; }
architecture="$(uname -m)"
topdir="$root/rpmbuild"
stage="$root/stage"
mkdir -p "$topdir"/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS} \
    "$stage/usr/lib/polypack-mcp" "$stage/usr/bin" \
    "$stage/usr/lib/systemd/system" "$stage/etc/default"

python3 -m pip install --no-deps --target "$stage/usr/lib/polypack-mcp" .
python3 -m pip install --target "$stage/usr/lib/polypack-mcp" \
    'polypack-db>=3.2.0' 'mcp>=1.9,<1.10' 'anyio>=4.5,<4.10'

cat > "$stage/usr/bin/polypack-mcp" <<'EOF'
#!/bin/sh
set -e
export PYTHONPATH=/usr/lib/polypack-mcp${PYTHONPATH:+:$PYTHONPATH}
exec /usr/bin/python3 -m polypack_mcp.server "$@"
EOF
chmod 0755 "$stage/usr/bin/polypack-mcp"

cat > "$stage/usr/lib/systemd/system/polypack-mcp.service" <<'EOF'
[Unit]
Description=Polypack MCP shared memory server
After=network.target

[Service]
Type=simple
User=polypack
Group=polypack
EnvironmentFile=-/etc/default/polypack-mcp
ExecStart=/usr/bin/polypack-mcp --transport streamable-http --host 127.0.0.1 --port $POLYPACK_MCP_PORT --store /var/lib/polypack-mcp
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

cat > "$stage/etc/default/polypack-mcp" <<'EOF'
# Change this value before restarting the service if port 8765 is occupied.
POLYPACK_MCP_PORT=8765
EOF

cat > "$topdir/SPECS/polypack-mcp.spec" <<EOF
Name: polypack-mcp
Version: $version
Release: 1%{?dist}
Summary: Persistent adaptive memory MCP server
License: Unspecified
BuildArch: $architecture
Requires: python3 >= 3.12
Requires: shadow-utils

%description
Polypack memory server for MCP clients such as Codex and Claude.

%prep

%build

%install
mkdir -p %{buildroot}
cp -a $stage/. %{buildroot}/

%pre
if ! getent passwd polypack >/dev/null; then
    useradd --system --user-group --home-dir /var/lib/polypack-mcp --no-create-home polypack
fi

%post
mkdir -p /var/lib/polypack-mcp
chown -R polypack:polypack /var/lib/polypack-mcp
chmod 0750 /var/lib/polypack-mcp
if command -v systemctl >/dev/null; then
    systemctl daemon-reload >/dev/null 2>&1 || true
    systemctl enable --now polypack-mcp.service >/dev/null 2>&1 || \
        echo "polypack-mcp installed; start it with: systemctl start polypack-mcp"
fi

%files
%config(noreplace) /etc/default/polypack-mcp
/usr/bin/polypack-mcp
/usr/lib/polypack-mcp
/usr/lib/systemd/system/polypack-mcp.service

%changelog
* $(date -u '+%a %b %d %Y') Polypack contributors - $version-1
- Release $version
EOF

mkdir -p "$output_dir"
rpmbuild --define "_topdir $topdir" --define "_rpmdir $output_dir" \
    --define "_builddir $topdir/BUILD" --define "_buildrootdir $topdir/BUILDROOT" \
    --define "_sourcedir $topdir/SOURCES" --define "_specdir $topdir/SPECS" \
    -bb "$topdir/SPECS/polypack-mcp.spec"
find "$output_dir" -mindepth 2 -type f -name '*.rpm' -exec cp {} "$output_dir" \;
