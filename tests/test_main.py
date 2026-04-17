"""Tests for the MCP server entry point.

Focused on the ``_resolve_repo_root`` helper that threads the
``serve --repo <X>`` CLI flag into every tool wrapper, and on the
set of tools that must be registered as async coroutines so the MCP
stdio event loop stays responsive during long-running operations.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys

import pytest

from code_review_graph import main as crg_main


class TestResolveRepoRoot:
    """Precedence rules for _resolve_repo_root (see #222 follow-up)."""

    @pytest.fixture(autouse=True)
    def _reset_default(self):
        """Save and restore the module-level default before/after each test."""
        original = crg_main._default_repo_root
        yield
        crg_main._default_repo_root = original

    def test_none_when_neither_is_set(self):
        crg_main._default_repo_root = None
        assert crg_main._resolve_repo_root(None) is None

    def test_empty_string_treated_as_unset(self):
        """Empty string from an MCP client should not shadow the --repo flag."""
        crg_main._default_repo_root = "/tmp/flag-repo"
        assert crg_main._resolve_repo_root("") == "/tmp/flag-repo"

    def test_flag_used_when_client_omits_repo_root(self):
        crg_main._default_repo_root = "/tmp/flag-repo"
        assert crg_main._resolve_repo_root(None) == "/tmp/flag-repo"

    def test_client_arg_wins_over_flag(self):
        crg_main._default_repo_root = "/tmp/flag-repo"
        assert crg_main._resolve_repo_root("/explicit") == "/explicit"

    def test_client_arg_used_when_no_flag(self):
        crg_main._default_repo_root = None
        assert crg_main._resolve_repo_root("/explicit") == "/explicit"

    def test_timed_tool_wrapper_preserves_signature(self):
        tool = crg_main.get_minimal_context_tool
        underlying = inspect.unwrap(
            getattr(tool, "fn", None)
            or getattr(tool, "_func", None)
            or getattr(tool, "func", None)
            or tool
        )
        signature = inspect.signature(underlying)
        assert list(signature.parameters.keys()) == [
            "task",
            "changed_files",
            "repo_root",
            "base",
        ]

    def test_configure_package_logging_targets_stderr(self):
        package_logger = logging.getLogger("code_review_graph")
        original_handlers = list(package_logger.handlers)
        original_level = package_logger.level
        original_propagate = package_logger.propagate

        try:
            package_logger.handlers = []
            package_logger.setLevel(logging.NOTSET)
            package_logger.propagate = True

            crg_main._configure_package_logging()

            stderr_handlers = [
                handler
                for handler in package_logger.handlers
                if isinstance(handler, logging.StreamHandler)
                and getattr(handler, "stream", None) is sys.stderr
            ]
            assert stderr_handlers, "expected a stderr stream handler for MCP logs"
            assert package_logger.level == logging.INFO
            assert package_logger.propagate is False

            before_count = len(package_logger.handlers)
            crg_main._configure_package_logging()
            assert len(package_logger.handlers) == before_count
        finally:
            package_logger.handlers = original_handlers
            package_logger.setLevel(original_level)
            package_logger.propagate = original_propagate


class TestLongRunningToolsAreAsync:
    """Long-running MCP tools must be registered as coroutines so the
    asyncio event loop stays responsive while the work runs in a
    background thread via ``asyncio.to_thread``. Without this, Windows
    MCP clients hang on ``build_or_update_graph_tool`` and
    ``embed_graph_tool`` — see #46, #136.
    """

    HEAVY_TOOLS = {
        "build_or_update_graph_tool",
        "run_postprocess_tool",
        "get_minimal_context_tool",
        "semantic_search_nodes_tool",
        "embed_graph_tool",
        "list_graph_stats_tool",
        "detect_changes_tool",
        "traverse_graph_tool",
        "cross_repo_search_tool",
        "generate_wiki_tool",
    }

    @pytest.mark.asyncio
    async def test_heavy_tools_are_coroutines(self):
        tools = await crg_main.mcp.get_tools()
        registered: dict[str, bool] = {}
        for name, tool in tools.items():
            if name not in self.HEAVY_TOOLS:
                continue
            # FastMCP 2.x stores the underlying Python function on the
            # tool wrapper; attribute name has varied but is typically
            # ``fn`` on FunctionTool. Fall back to a few candidates.
            fn = (
                getattr(tool, "fn", None)
                or getattr(tool, "_func", None)
                or getattr(tool, "func", None)
                or tool
            )
            registered[name] = asyncio.iscoroutinefunction(fn)

        missing = self.HEAVY_TOOLS - registered.keys()
        assert not missing, f"heavy tool(s) not registered at all: {missing}"

        not_async = [name for name, is_async in registered.items() if not is_async]
        assert not not_async, (
            f"these tools must be async but were registered as sync, "
            f"which will hang the stdio event loop on Windows: {not_async}"
        )

    @pytest.mark.asyncio
    async def test_heavy_tool_source_uses_to_thread(self):
        """Defense in depth: heavy wrappers should still show explicit
        offloading logic in source so we don't silently regress Windows MCP
        responsiveness."""
        for tool_name in self.HEAVY_TOOLS:
            fn = getattr(crg_main, tool_name, None)
            assert fn is not None, f"{tool_name} not found on module"
            underlying = inspect.unwrap(getattr(fn, "fn", None) or fn)
            source = inspect.getsource(underlying)
            if tool_name == "run_postprocess_tool":
                assert "asyncio.to_thread" in source
                assert "if communities:" in source
                continue
            assert "asyncio.to_thread" in source, (
                f"{tool_name} must call asyncio.to_thread to offload its "
                f"blocking work; otherwise Windows MCP clients will hang. "
                f"See #46, #136."
            )


class TestRunPostprocessToolExecutionMode:
    @pytest.mark.asyncio
    async def test_communities_path_runs_inline(self, monkeypatch):
        tool = crg_main.run_postprocess_tool
        underlying = inspect.unwrap(getattr(tool, "fn", None) or tool)

        called: dict[str, object] = {}

        def fake_run_postprocess(**kwargs):
            called["kwargs"] = kwargs
            return {"status": "ok", "mode": "inline"}

        async def fail_to_thread(*args, **kwargs):
            raise AssertionError("communities path should not use asyncio.to_thread")

        monkeypatch.setattr(crg_main, "run_postprocess", fake_run_postprocess)
        monkeypatch.setattr(crg_main.asyncio, "to_thread", fail_to_thread)

        result = await underlying(
            flows=False,
            communities=True,
            fts=False,
            repo_root="/tmp/repo",
        )

        assert result == {"status": "ok", "mode": "inline"}
        assert called["kwargs"] == {
            "flows": False,
            "communities": True,
            "fts": False,
            "repo_root": "/tmp/repo",
        }

    @pytest.mark.asyncio
    async def test_non_community_path_stays_threaded(self, monkeypatch):
        tool = crg_main.run_postprocess_tool
        underlying = inspect.unwrap(getattr(tool, "fn", None) or tool)

        called: dict[str, object] = {}

        def fake_run_postprocess(**kwargs):
            called["run_postprocess_kwargs"] = kwargs
            return {"status": "ok", "mode": "threaded"}

        async def fake_to_thread(func, /, *args, **kwargs):
            called["to_thread_func"] = func
            called["to_thread_kwargs"] = kwargs
            return func(*args, **kwargs)

        monkeypatch.setattr(crg_main, "run_postprocess", fake_run_postprocess)
        monkeypatch.setattr(crg_main.asyncio, "to_thread", fake_to_thread)

        result = await underlying(
            flows=True,
            communities=False,
            fts=True,
            repo_root="/tmp/repo",
        )

        assert result == {"status": "ok", "mode": "threaded"}
        assert called["to_thread_func"] is fake_run_postprocess
        assert called["to_thread_kwargs"] == {
            "flows": True,
            "communities": False,
            "fts": True,
            "repo_root": "/tmp/repo",
        }
        assert called["run_postprocess_kwargs"] == called["to_thread_kwargs"]
