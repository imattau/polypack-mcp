#!/usr/bin/env bash
set -euo pipefail

version="${1:?usage: build.sh VERSION OUTPUT_DIR}"
output_dir="${2:?usage: build.sh VERSION OUTPUT_DIR}"
root="$(mktemp -d)"
trap 'rm -rf "$root"' EXIT

architecture="$(dpkg --print-architecture)"
stage="$root/polypack-mcp_${version}_${architecture}"
mkdir -p "$stage/DEBIAN" "$stage/usr/lib/polypack-mcp" "$stage/usr/bin" "$stage/lib/systemd/system" "$stage/etc/default"
python3 -m pip install --no-deps --target "$stage/usr/lib/polypack-mcp" .
python3 -m pip install --target "$stage/usr/lib/polypack-mcp" 'polypack-db>=3.3.1' 'mcp>=1.9,<1.10' 'anyio>=4.5,<4.10'

cat > "$stage/usr/bin/polypack-mcp" <<'EOF'
#!/bin/sh
set -e
export PYTHONPATH=/usr/lib/polypack-mcp${PYTHONPATH:+:$PYTHONPATH}
exec /usr/bin/python3 -m polypack_mcp.server "$@"
EOF
chmod 0755 "$stage/usr/bin/polypack-mcp"

cat > "$stage/DEBIAN/control" <<EOF
Package: polypack-mcp
Version: $version
Section: utils
Priority: optional
Architecture: $architecture
Depends: python3 (>= 3.12), adduser
Suggests: systemd
Maintainer: Polypack contributors
Description: Persistent adaptive memory MCP server
 Polypack memory server for MCP clients such as Codex and Claude.
EOF

cat > "$stage/lib/systemd/system/polypack-mcp.service" <<'EOF'
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
echo /etc/default/polypack-mcp > "$stage/DEBIAN/conffiles"

cat > "$stage/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e

if ! getent passwd polypack >/dev/null; then
    adduser --system --group --home /var/lib/polypack-mcp --no-create-home polypack
fi
mkdir -p /var/lib/polypack-mcp
chown -R polypack:polypack /var/lib/polypack-mcp
chmod 0750 /var/lib/polypack-mcp

if command -v systemctl >/dev/null && systemctl daemon-reload >/dev/null 2>&1; then
    if systemctl is-active --quiet polypack-mcp.service; then
        systemctl restart polypack-mcp.service >/dev/null 2>&1 || \
            echo "polypack-mcp updated; restart it with: sudo systemctl restart polypack-mcp"
    else
        systemctl enable --now polypack-mcp.service >/dev/null 2>&1 || \
            echo "polypack-mcp installed; start it with: sudo systemctl start polypack-mcp"
    fi
else
    echo "polypack-mcp installed; systemd was not detected, so the service was not started"
fi
exit 0
EOF
chmod 0755 "$stage/DEBIAN/postinst"

mkdir -p "$output_dir"
dpkg-deb --build "$stage" "$output_dir/polypack-mcp_${version}_${architecture}.deb"
