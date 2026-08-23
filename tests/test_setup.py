from polypack_mcp.setup import print_client_config


def test_client_config_uses_shared_streamable_http_endpoint(capsys):
    print_client_config(8123)
    output = capsys.readouterr().out
    assert "http://127.0.0.1:8123/mcp/" in output
    assert "mcp_servers.polypack" in output
    assert '"mcpServers"' in output
