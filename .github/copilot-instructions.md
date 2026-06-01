# Copilot Instructions

## Session Start
- Read `0.dev-matrix/STATE.md`, `0.dev-matrix/INDEX.md`, the newest `0.dev-matrix/AI-HANDOFF.md`, and the task board before planning.
- If `0.dev-matrix/resume-work.ps1` exists, run it before coding.

## Roo Code Index Bridge MCP
Use the global MCP server `roo-code-index-bridge` as the default semantic retrieval surface before falling back to grep or regex.
Do not register legacy `roo-index-bridge` alongside it.

Before planning or coding in this repo, read `0.dev-matrix/INDEX.md` and the newest `0.dev-matrix/AI-HANDOFF.md`.

- `roo-code-index-search`: primary semantic search - pass `workspace_path="D:/Github/code-review-graph"`
- `roo-code-index-resolve-collection`: verify workspace mapping when results look suspicious
- `roo-code-index-health`: check index health on unfamiliar repos

Preferred retrieval stack for code work:

1. `roo-code-index-bridge_roo-code-index-search` for broad discovery
2. Graphify `graphify_query_graph`, `graphify_graph_stats`, `graphify_get_community`, `graphify_god_nodes`, or `graphify_shortest_path` for structural orientation
3. code-review-graph `code-review-graph_get_minimal_context_tool`, `code-review-graph_get_impact_radius_tool`, `code-review-graph_get_affected_flows_tool`, or `code-review-graph_query_graph_tool` (always pass `repo_root`)
4. grep or regex for exact confirmation

Use only the exact MCP tool names listed above, including the required prefixes and suffixes.

### Knowledge Ledger Gate
Before non-trivial edits:

1. use Graphify or `graphify-out/GRAPH_REPORT.md` to map the owning structure
2. use code-review-graph to assess blast radius and impacted flows
3. return a short `CONTEXT AUDIT` before implementation with:
    - `Slice:`
    - `Files:`
    - `Dependencies:`
    - `Test first:`
    - `Proof:`

### Test-First Gate
For behavior changes, bug fixes, or refactors that change behavior:

1. write or update the narrow automated test first
2. run it and confirm it fails for the expected reason
3. implement the minimum change required
4. rerun the same test until it passes
5. only then widen to the next narrow validation

For parallel isolated subtasks, use `agent-delegator` (`delegate_task` / `batch_tasks`) - not one-liners.

Validation:
```powershell
node D:\Github\tools\roo-index-smoke.mjs --workspace D:\Github\code-review-graph
node D:\Github\tools\roo-index-sync-mcp.mjs --all --apply
```

## Close-Day
- Update `0.dev-matrix/AI-HANDOFF.md` with `Changed:`, `Pending:`, `Verified:`, `Operational proof:`, `Continue from:`, `Next step:`, `Technical debt:`, and `Blockers:`.
- Keep `0.dev-matrix/LAUNCH_CHECKLIST.md` truthful.
