# TASK — code-review-graph

> Scope: semantic code-review MCP and graph engine stabilization

---

## ACTIVE TASKS

| ID | Task | Type | Owner | Status |
|----|------|------|-------|--------|
| CRG-201 | Validate the Windows async `asyncio.to_thread` tool path across the heavy graph operations and keep the pytest slice green | Test | AI | 🔲 TODO |
| CRG-202 | Clean repo-safe rollout diffs and commit the current working MCP stage without losing local-only evidence | Maintenance | AI | 🔲 TODO |
| CRG-203 | Add framework-aware dead-code and community-splitting regression tests for large real repos | Test | AI | 🔲 TODO |
| CRG-204 | Validate multi-format exports (GraphML, Neo4j, SVG, Obsidian) against a real graph database build | Feature | AI | 🔲 TODO |
| CRG-205 | Extend language and parser verification for the newest added language surfaces and edge-case files | Compatibility | AI | 🔲 TODO |

## COMPLETED TASKS

| ID | Task | Completed | Evidence |
|----|------|-----------|---------|
| CRG-001 | Reach MCP working stage with repo-local rollout repaired across D:\Github | 2026-04-16 | 30/30 smoke tests and preserved repo-specific MCP servers |
| CRG-002 | Ship Windows stdio/runtime fixes so the MCP can run reliably in the portfolio | 2026-04-16 | STATE.md handoff entry + passing smoke retest |