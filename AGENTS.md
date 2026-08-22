# Project instructions

## Polypack memory

Use the connected Polypack MCP when working on this repository:

- Before tasks involving prior decisions, project history, preferences, or accumulated context, call `mcp__polypack__memory_recall` with context `polypack-mcp`.
- Store durable architectural decisions, important discoveries, workflows, and user preferences with `mcp__polypack__memory_store` using context `polypack-mcp`.
- After completing a meaningful task, store a concise summary of the outcome when it is likely to help future work.
- Use `mcp__polypack__memory_context` when a bounded working-memory set is more useful than a keyword search.
- Do not store secrets, credentials, personal sensitive data, transient debugging output, or trivial conversation.
- If Polypack is unavailable or a call fails, continue the task and mention the limitation only when it affects the result.
