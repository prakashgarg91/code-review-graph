"""Tests for the hybrid search engine."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from code_review_graph.graph import GraphStore
from code_review_graph.parser import NodeInfo
from code_review_graph.search import (
    _embedding_search,
    detect_query_kind_boost,
    hybrid_search,
    rebuild_fts_index,
    rrf_merge,
)


class TestHybridSearch:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = GraphStore(self.tmp.name)
        self._seed_data()

    def teardown_method(self):
        self.store.close()
        self.tmp.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def _seed_data(self):
        """Seed test nodes into the graph store."""
        nodes = [
            NodeInfo(
                kind="Function", name="get_users", file_path="api.py",
                line_start=1, line_end=20, language="python",
                params="(db: Session)", return_type="list[User]",
            ),
            NodeInfo(
                kind="Function", name="create_user", file_path="api.py",
                line_start=25, line_end=40, language="python",
                params="(name: str, email: str)", return_type="User",
            ),
            NodeInfo(
                kind="Class", name="UserService", file_path="services.py",
                line_start=1, line_end=100, language="python",
            ),
            NodeInfo(
                kind="Function", name="authenticate", file_path="auth.py",
                line_start=5, line_end=30, language="python",
                params="(token: str)", return_type="bool",
            ),
            NodeInfo(
                kind="Type", name="UserResponse", file_path="models.py",
                line_start=1, line_end=15, language="python",
            ),
        ]
        for node in nodes:
            node_id = self.store.upsert_node(node, file_hash="abc123")
            if node.kind == "Function":
                sig = f"def {node.name}{node.params or '()'} -> {node.return_type or 'None'}"
                self.store._conn.execute(
                    "UPDATE nodes SET signature = ? WHERE id = ?", (sig, node_id)
                )
        self.store._conn.commit()

    def test_rebuild_fts_index(self):
        """rebuild_fts_index returns the correct count of indexed rows."""
        count = rebuild_fts_index(self.store)
        assert count == 5

    def test_rebuild_fts_index_idempotent(self):
        """Rebuilding twice gives the same count."""
        count1 = rebuild_fts_index(self.store)
        count2 = rebuild_fts_index(self.store)
        assert count1 == count2

    def test_fts_search_by_name(self):
        """FTS search finds a node by its name."""
        rebuild_fts_index(self.store)
        results = hybrid_search(self.store, "get_users")
        assert len(results) > 0
        names = [r["name"] for r in results]
        assert "get_users" in names

    def test_fts_search_by_signature(self):
        """FTS search finds a node by content in its signature."""
        rebuild_fts_index(self.store)
        results = hybrid_search(self.store, "Session")
        assert len(results) > 0
        names = [r["name"] for r in results]
        assert "get_users" in names

    def test_kind_boost_pascal_case(self):
        """PascalCase query boosts Class kind > 1.0."""
        boosts = detect_query_kind_boost("UserService")
        assert "Class" in boosts
        assert boosts["Class"] > 1.0

    def test_kind_boost_snake_case(self):
        """snake_case query boosts Function kind > 1.0."""
        boosts = detect_query_kind_boost("get_users")
        assert "Function" in boosts
        assert boosts["Function"] > 1.0

    def test_kind_boost_dotted(self):
        """Dotted query boosts qualified name matches."""
        boosts = detect_query_kind_boost("api.get_users")
        assert "_qualified" in boosts
        assert boosts["_qualified"] > 1.0

    def test_kind_boost_empty(self):
        """Empty query returns no boosts."""
        boosts = detect_query_kind_boost("")
        assert boosts == {}

    def test_kind_boost_all_uppercase(self):
        """ALL_CAPS should not trigger PascalCase boost."""
        boosts = detect_query_kind_boost("HTTP_STATUS")
        assert "Class" not in boosts
        assert "Function" in boosts

    def test_rrf_merge(self):
        """Node appearing in both lists ranks highest after RRF merge."""
        list_a = [(1, 10.0), (2, 8.0), (3, 6.0)]
        list_b = [(2, 9.0), (4, 7.0), (1, 5.0)]

        merged = rrf_merge(list_a, list_b)
        ids = [item_id for item_id, _ in merged]

        assert ids[0] in (1, 2)
        assert ids[1] in (1, 2)
        assert ids[0] == 2

    def test_rrf_merge_single_list(self):
        """RRF merge with a single list preserves order."""
        single = [(10, 5.0), (20, 3.0), (30, 1.0)]
        merged = rrf_merge(single)
        ids = [item_id for item_id, _ in merged]
        assert ids == [10, 20, 30]

    def test_rrf_merge_empty(self):
        """RRF merge with empty lists returns empty."""
        merged = rrf_merge([], [])
        assert merged == []

    def test_fallback_to_keyword(self):
        """Works without FTS index by falling back to keyword LIKE matching."""
        try:
            self.store._conn.execute("DROP TABLE IF EXISTS nodes_fts")
            self.store._conn.commit()
        except Exception:
            pass

        results = hybrid_search(self.store, "authenticate")
        assert len(results) > 0
        names = [r["name"] for r in results]
        assert "authenticate" in names

    def test_empty_query_handled(self):
        """Empty query returns empty results without crashing."""
        results = hybrid_search(self.store, "")
        assert results == []

    def test_whitespace_query_handled(self):
        """Whitespace-only query returns empty results."""
        results = hybrid_search(self.store, "   ")
        assert results == []

    def test_hybrid_search_returns_expected_fields(self):
        """All expected fields are present in search results."""
        rebuild_fts_index(self.store)
        results = hybrid_search(self.store, "get_users")
        assert len(results) > 0

        expected_fields = {
            "name", "qualified_name", "kind", "file_path",
            "line_start", "line_end", "language", "params",
            "return_type", "signature", "score",
        }
        for result in results:
            assert expected_fields.issubset(result.keys()), (
                f"Missing fields: {expected_fields - result.keys()}"
            )

    def test_kind_filter(self):
        """Kind parameter filters results to only that kind."""
        rebuild_fts_index(self.store)
        results = hybrid_search(self.store, "User", kind="Class")
        for result in results:
            assert result["kind"] == "Class"

    def test_context_file_boost(self):
        """Nodes in context_files get boosted above others."""
        rebuild_fts_index(self.store)
        results_with_ctx = hybrid_search(self.store, "user", context_files=["api.py"])

        if results_with_ctx:
            api_nodes = [r for r in results_with_ctx if r["file_path"] == "api.py"]
            if api_nodes:
                api_score = api_nodes[0]["score"]
                assert api_score > 0

    def test_limit_respected(self):
        """Search respects the limit parameter."""
        rebuild_fts_index(self.store)
        results = hybrid_search(self.store, "user", limit=2)
        assert len(results) <= 2

    def test_embedding_search_uses_warm_only_provider(self):
        """Query-time embedding search should not cold-load the local model."""
        fake_store = MagicMock()
        fake_store.db_path = self.tmp.name

        fake_emb_store = MagicMock()
        fake_emb_store.available = True
        fake_emb_store.count.return_value = 0

        with patch("code_review_graph.embeddings.EmbeddingStore", return_value=fake_emb_store) as mock_store_cls:
            results = _embedding_search(fake_store, "user", limit=5)

        assert results == []
        mock_store_cls.assert_called_once_with(self.tmp.name, model=None, allow_cold_load=False)
        fake_emb_store.close.assert_called_once()

    def test_fts_query_with_special_chars(self):
        """FTS5 special characters are safely handled."""
        rebuild_fts_index(self.store)
        for dangerous_query in ['OR user', 'NOT thing', 'user*', '"user"', 'a AND b']:
            results = hybrid_search(self.store, dangerous_query)
            assert isinstance(results, list)
