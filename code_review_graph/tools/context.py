"""Tool: get_minimal_context — ultra-compact context for token-efficient workflows."""

from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

from ..incremental import find_project_root, get_db_path
from ._common import _validate_repo_root, compact_response

logger = logging.getLogger(__name__)

_MAX_CHANGED_FILES_FOR_DEEP_RISK = 12
_MINIMAL_CONTEXT_GIT_TIMEOUT = 2


def _minimal_context_git_env() -> dict[str, str]:
    """Return a non-interactive git environment safe for stdio MCP servers."""
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GCM_INTERACTIVE", "Never")
    env.setdefault("GIT_PAGER", "cat")
    env.setdefault("PAGER", "cat")
    return env


def _resolve_root(repo_root: str | None) -> Path:
    """Resolve and validate the target project root."""
    if repo_root:
        return _validate_repo_root(Path(repo_root))

    root = find_project_root()
    if root is None:
        raise ValueError("Could not determine project root for get_minimal_context")
    return root


def _open_context_connection(root: Path) -> sqlite3.Connection:
    """Open the graph database in read-only mode for minimal-context reads."""
    db_path = get_db_path(root)
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _query_single_int(conn: sqlite3.Connection, sql: str) -> int:
    """Run a scalar integer query, returning 0 when the table is absent."""
    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0]) if row else 0


def _query_top_names(conn: sqlite3.Connection, sql: str) -> list[str]:
    """Run a single-column name query, returning an empty list if unavailable."""
    try:
        rows = conn.execute(sql).fetchall()
    except sqlite3.OperationalError:
        return []
    return [str(row[0]) for row in rows if row and row[0]]


def _summarize_lightweight_risk(root: Path, files: list[str]) -> tuple[str, float, list[str], int]:
    """Compute a cheap risk summary using impact radius instead of full review analysis."""
    from ..graph import GraphStore

    store = GraphStore(get_db_path(root))
    try:
        abs_files = [str(root / file_path) for file_path in files]
        impact = store.get_impact_radius(abs_files, max_depth=1)
        impacted_count = len(impact["impacted_nodes"])
        changed_nodes = impact["changed_nodes"]
        top_affected = [node.name for node in changed_nodes[:5]]
        if not top_affected:
            top_affected = [node.name for node in impact["impacted_nodes"][:5]]

        if impacted_count > 20:
            return "high", 0.8, top_affected, impacted_count
        if impacted_count > 5:
            return "medium", 0.5, top_affected, impacted_count
        if changed_nodes:
            return "low", 0.2, top_affected, impacted_count
        return "unknown", 0.0, top_affected, impacted_count
    finally:
        store.close()


def _detect_changed_files(root: Path, base: str, changed_files: list[str] | None) -> tuple[list[str], str]:
    """Return changed files for minimal-context without a separate pre-check."""
    if changed_files is not None:
        return changed_files, "provided"

    diff_started = time.perf_counter()
    logger.info(
        "get_minimal_context.git_diff_start repo_root=%s timeout_s=%d",
        root,
        _MINIMAL_CONTEXT_GIT_TIMEOUT,
    )
    try:
        result = subprocess.run(
            [
                "git",
                "-c", "core.pager=cat",
                "-c", "pager.diff=false",
                "-c", "interactive.diffFilter=false",
                "diff",
                "--name-only",
                "--no-ext-diff",
                "--no-textconv",
                base,
                "--",
            ],
            capture_output=True,
            text=True,
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            env=_minimal_context_git_env(),
            timeout=_MINIMAL_CONTEXT_GIT_TIMEOUT,
        )
        if result.returncode == 0:
            files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            logger.info(
                "get_minimal_context.git_diff_done repo_root=%s returncode=%d changed_files=%d duration_ms=%d",
                root,
                result.returncode,
                len(files),
                int((time.perf_counter() - diff_started) * 1000),
            )
            if files:
                return files, "git-diff"
            return [], "git-diff-empty"
        logger.info(
            "get_minimal_context.git_diff_done repo_root=%s returncode=%d changed_files=0 duration_ms=%d",
            root,
            result.returncode,
            int((time.perf_counter() - diff_started) * 1000),
        )
        return [], f"git-diff-exit-{result.returncode}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.info(
            "get_minimal_context.git_diff_timeout repo_root=%s duration_ms=%d",
            root,
            int((time.perf_counter() - diff_started) * 1000),
        )
        return [], "git-diff-timeout"
    except (ImportError, OSError, ValueError, subprocess.SubprocessError):
        logger.debug("Change detection failed in get_minimal_context", exc_info=True)

    return [], "none"


def get_minimal_context(
    task: str = "",
    changed_files: list[str] | None = None,
    repo_root: str | None = None,
    base: str = "HEAD~1",
) -> dict[str, Any]:
    """Return minimum context an agent needs to start any task (~100 tokens).

    Combines graph stats, top communities, top flows, risk score,
    and suggested next tools into an ultra-compact response.

    Args:
        task: Natural language description of what the agent is doing
              (e.g. "review PR #42", "debug login timeout").
        changed_files: Explicit changed files. Auto-detected if None.
        repo_root: Repository root path. Auto-detected if None.
        base: Git ref for diff comparison.
    """
    root = _resolve_root(repo_root)
    conn_started = time.perf_counter()
    conn = _open_context_connection(root)
    logger.info(
        "get_minimal_context.open_db repo_root=%s duration_ms=%d",
        root,
        int((time.perf_counter() - conn_started) * 1000),
    )

    try:
        # 1. Quick stats
        stats_started = time.perf_counter()
        total_nodes = _query_single_int(conn, "SELECT COUNT(*) FROM nodes")
        total_edges = _query_single_int(conn, "SELECT COUNT(*) FROM edges")
        files_count = _query_single_int(
            conn,
            "SELECT COUNT(*) FROM nodes WHERE kind = 'File'",
        )
        logger.info(
            "get_minimal_context.stats repo_root=%s duration_ms=%d total_nodes=%d total_edges=%d files=%d",
            root,
            int((time.perf_counter() - stats_started) * 1000),
            total_nodes,
            total_edges,
            files_count,
        )

        # 2. Risk from changed files
        risk = "unknown"
        risk_score = 0.0
        top_affected: list[str] = []
        test_gap_count = 0
        changed_file_count = 0
        risk_analysis_mode = "none"
        detect_started = time.perf_counter()
        files, change_source = _detect_changed_files(root, base, changed_files)
        changed_file_count = len(files)
        logger.info(
            "get_minimal_context.change_detect repo_root=%s source=%s changed_files=%d duration_ms=%d",
            root,
            change_source,
            changed_file_count,
            int((time.perf_counter() - detect_started) * 1000),
        )

        if files:
            risk_started = time.perf_counter()
            try:
                if changed_file_count > _MAX_CHANGED_FILES_FOR_DEEP_RISK:
                    risk_analysis_mode = "skipped"
                    logger.info(
                        "get_minimal_context.fast_path repo_root=%s changed_files=%d threshold=%d",
                        root,
                        changed_file_count,
                        _MAX_CHANGED_FILES_FOR_DEEP_RISK,
                    )
                else:
                    analysis_started = time.perf_counter()
                    risk, risk_score, top_affected, impacted_count = _summarize_lightweight_risk(root, files)
                    risk_analysis_mode = "light"
                    logger.info(
                        "get_minimal_context.light_risk repo_root=%s changed_files=%d duration_ms=%d risk_score=%.2f impacted_nodes=%d",
                        root,
                        changed_file_count,
                        int((time.perf_counter() - analysis_started) * 1000),
                        risk_score,
                        impacted_count,
                    )
            except (
                ImportError, OSError, ValueError,
                sqlite3.Error, subprocess.SubprocessError,
            ):
                logger.debug("Risk analysis failed in get_minimal_context", exc_info=True)
            finally:
                logger.info(
                    "get_minimal_context.change_scan repo_root=%s changed_files=%d mode=%s duration_ms=%d",
                    root,
                    changed_file_count,
                    risk_analysis_mode,
                    int((time.perf_counter() - risk_started) * 1000),
                )

        # 3. Top 3 communities
        communities_started = time.perf_counter()
        communities = _query_top_names(
            conn,
            "SELECT name FROM communities ORDER BY size DESC LIMIT 3",
        )
        logger.info(
            "get_minimal_context.communities repo_root=%s duration_ms=%d count=%d",
            root,
            int((time.perf_counter() - communities_started) * 1000),
            len(communities),
        )

        # 4. Top 3 critical flows
        flows_started = time.perf_counter()
        flows = _query_top_names(
            conn,
            "SELECT name FROM flows ORDER BY criticality DESC LIMIT 3",
        )
        logger.info(
            "get_minimal_context.flows repo_root=%s duration_ms=%d count=%d",
            root,
            int((time.perf_counter() - flows_started) * 1000),
            len(flows),
        )

        # 5. Suggest next tools based on task keywords
        task_lower = task.lower()
        if any(word in task_lower for word in ("review", "pr", "merge", "diff")):
            suggestions = ["detect_changes", "get_affected_flows", "get_review_context"]
        elif any(word in task_lower for word in ("debug", "bug", "error", "fix")):
            suggestions = ["semantic_search_nodes", "query_graph", "get_flow"]
        elif any(word in task_lower for word in ("refactor", "rename", "dead", "clean")):
            suggestions = ["refactor", "find_large_functions", "get_architecture_overview"]
        elif any(word in task_lower for word in ("onboard", "understand", "explore", "arch")):
            suggestions = [
                "get_architecture_overview", "list_communities", "list_flows",
            ]
        else:
            suggestions = [
                "detect_changes", "semantic_search_nodes", "get_architecture_overview",
            ]

        summary_parts = [
            f"{total_nodes} nodes, {total_edges} edges across {files_count} files.",
        ]
        if changed_file_count:
            summary_parts.append(f"{changed_file_count} changed file(s) detected.")
        if risk != "unknown":
            summary_parts.append(f"Risk: {risk} ({risk_score:.2f}).")
        elif risk_analysis_mode == "skipped":
            summary_parts.append("Deep risk analysis skipped for fast path.")
        elif risk_analysis_mode == "light":
            summary_parts.append("Lightweight risk heuristic used for fast path.")
        if test_gap_count:
            summary_parts.append(f"{test_gap_count} test gaps.")

        return compact_response(
            summary=" ".join(summary_parts),
            key_entities=top_affected or None,
            risk=risk,
            communities=communities or None,
            flows_affected=flows or None,
            next_tool_suggestions=suggestions,
        )
    finally:
        close_started = time.perf_counter()
        conn.close()
        logger.info(
            "get_minimal_context.close_db repo_root=%s duration_ms=%d",
            root,
            int((time.perf_counter() - close_started) * 1000),
        )
