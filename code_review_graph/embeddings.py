"""Vector embedding support for semantic code search.

Supports multiple providers:
1. Local (sentence-transformers) - Private, fast, offline.
2. Ollama - Private, local HTTP embeddings for tagged models like nomic-embed-text-v2-moe.
3. Google Gemini - High-quality, cloud-based. Requires explicit opt-in.
4. MiniMax (embo-01) - High-quality 1536-dim cloud embeddings. Requires MINIMAX_API_KEY.
"""

from __future__ import annotations

import hashlib
import json
import importlib.util
import logging
import os
import sqlite3
import struct
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any

from .graph import GraphNode, GraphStore, node_to_dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider Interface and Implementations
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (may use a different task type than indexing)."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


LOCAL_DEFAULT_MODEL = "all-MiniLM-L6-v2"
OLLAMA_DEFAULT_MODEL = "nomic-embed-text-v2-moe:latest"
OLLAMA_DEFAULT_URL = "http://localhost:11434"


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str | None = None) -> None:
        if not _check_available():
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install code-review-graph[embeddings]"
            )
        self._model_name = model_name or os.environ.get(
            "CRG_EMBEDDING_MODEL", LOCAL_DEFAULT_MODEL
        )
        self._model = None  # Lazy-loaded

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self._model_name,
                    trust_remote_code=True,
                    model_kwargs={"trust_remote_code": True},
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install code-review-graph[embeddings]"
                )
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        model = self._get_model()
        return model.get_sentence_embedding_dimension()

    @property
    def name(self) -> str:
        return f"local:{self._model_name}"


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model_name = model_name or os.environ.get(
            "CRG_EMBEDDING_MODEL", OLLAMA_DEFAULT_MODEL
        )
        self._base_url = (
            base_url
            or os.environ.get("CRG_OLLAMA_URL")
            or os.environ.get("OLLAMA_HOST")
            or OLLAMA_DEFAULT_URL
        ).rstrip("/")
        dim_env = os.environ.get("CRG_OLLAMA_EMBEDDING_DIMENSIONS", "").strip()
        self._dimension = int(dim_env) if dim_env else None
        self._keep_alive = os.environ.get("CRG_OLLAMA_KEEP_ALIVE", "10m")

    def _call_api(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []

        payload: dict[str, Any] = {
            "model": self._model_name,
            "input": [f"{prefix}{text}" for text in texts],
            "truncate": True,
        }
        if self._dimension is not None:
            payload["dimensions"] = self._dimension
        if self._keep_alive:
            payload["keep_alive"] = self._keep_alive

        req = urllib.request.Request(
            f"{self._base_url}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Unable to reach Ollama at {self._base_url}: {exc}"
            ) from exc

        embeddings = body.get("embeddings", [])
        if embeddings and self._dimension is None:
            self._dimension = len(embeddings[0])
        return embeddings

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._call_api(texts, "search_document: ")

    def embed_query(self, text: str) -> list[float]:
        return self._call_api([text], "search_query: ")[0]

    @property
    def dimension(self) -> int:
        return self._dimension or 768

    @property
    def name(self) -> str:
        return f"ollama:{self._model_name}"



class GoogleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "gemini-embedding-001") -> None:
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            self.model = model
            self._dimension: int | None = None
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Run: pip install code-review-graph[google-embeddings]"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        batch_size = 100
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._call_with_retry(
                lambda b=batch: self._client.models.embed_content(
                    model=self.model,
                    contents=b,
                    config={"task_type": "RETRIEVAL_DOCUMENT"},
                )
            )
            results.extend([e.values for e in response.embeddings])
        if self._dimension is None and results:
            self._dimension = len(results[0])
        return results

    @staticmethod
    def _call_with_retry(fn, max_retries: int = 3):
        """Call fn with exponential backoff on transient API errors."""
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                # Retry on rate-limit (429) or server errors (5xx)
                err_str = str(e)
                is_retryable = "429" in err_str or "500" in err_str or "503" in err_str
                if not is_retryable or attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning("Gemini API error (attempt %d/%d), retrying in %ds: %s",
                               attempt + 1, max_retries, wait, e)
                time.sleep(wait)

    def embed_query(self, text: str) -> list[float]:
        response = self._call_with_retry(
            lambda: self._client.models.embed_content(
                model=self.model,
                contents=[text],
                config={"task_type": "RETRIEVAL_QUERY"},
            )
        )
        vec = response.embeddings[0].values
        if self._dimension is None:
            self._dimension = len(vec)
        return vec

    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        # Default for gemini-embedding-001; updated dynamically after first call
        return 768

    @property
    def name(self) -> str:
        return f"google:{self.model}"


class MiniMaxEmbeddingProvider(EmbeddingProvider):
    """MiniMax embo-01 embedding provider (1536 dimensions).

    Uses the MiniMax Embeddings API (https://api.minimax.io/v1/embeddings)
    with the embo-01 model. Requires the MINIMAX_API_KEY environment variable.
    """

    _ENDPOINT = "https://api.minimax.io/v1/embeddings"
    _MODEL = "embo-01"
    _DIMENSION = 1536

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def _call_api(self, texts: list[str], task_type: str) -> list[list[float]]:
        import json as _json
        import urllib.request

        payload = _json.dumps({
            "model": self._MODEL,
            "texts": texts,
            "type": task_type,
        }).encode("utf-8")

        req = urllib.request.Request(
            self._ENDPOINT,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                import ssl
                _ssl_ctx = ssl.create_default_context()
                with urllib.request.urlopen(req, timeout=60, context=_ssl_ctx) as resp:  # nosec B310
                    body = _json.loads(resp.read().decode("utf-8"))

                base_resp = body.get("base_resp", {})
                if base_resp.get("status_code", 0) != 0:
                    raise RuntimeError(
                        f"MiniMax API error: {base_resp.get('status_msg', 'unknown')}"
                    )

                return body["vectors"]
            except Exception as e:
                err_str = str(e)
                is_retryable = "429" in err_str or "500" in err_str or "503" in err_str
                if not is_retryable or attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "MiniMax API error (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, max_retries, wait, e,
                )
                time.sleep(wait)

        return []  # unreachable, but keeps mypy happy

    def embed(self, texts: list[str]) -> list[list[float]]:
        batch_size = 100
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.extend(self._call_api(batch, "db"))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._call_api([text], "query")[0]

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    @property
    def name(self) -> str:
        return f"minimax:{self._MODEL}"


CLOUD_PROVIDERS = {"google", "minimax"}


def _warn_cloud_egress(provider_name: str) -> None:
    """Print a stderr warning before a cloud embedding provider is used.

    The warning is suppressed when ``CRG_ACCEPT_CLOUD_EMBEDDINGS=1`` is
    set in the environment, so scripted / CI workloads can acknowledge
    once and move on. Use stderr (never stdin/input) to stay compatible
    with the MCP stdio transport — anything we write to stdout would
    corrupt the JSON-RPC stream. See: #174
    """
    if os.environ.get("CRG_ACCEPT_CLOUD_EMBEDDINGS", "").strip() == "1":
        return
    print(
        f"\n⚠️  code-review-graph: about to embed code via the '{provider_name}' "
        "cloud provider.\n"
        "    Your source code (function names, docstrings, file paths) will be "
        "sent to an external API.\n"
        "    This is necessary for semantic search with the cloud provider you "
        "selected.\n"
        "    To skip this warning in future runs, set "
        "CRG_ACCEPT_CLOUD_EMBEDDINGS=1 in your environment.\n"
        "    To stay fully offline, use the default 'local' provider instead "
        "(no API key needed).\n",
        file=sys.stderr,
    )


def _get_ollama_base_url() -> str:
    return (
        os.environ.get("CRG_OLLAMA_URL")
        or os.environ.get("OLLAMA_HOST")
        or OLLAMA_DEFAULT_URL
    ).rstrip("/")


@lru_cache(maxsize=8)
def _ollama_list_models(base_url: str) -> tuple[str, ...]:
    req = urllib.request.Request(f"{base_url}/api/tags")
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:  # nosec B310
            body = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return ()

    models: list[str] = []
    for model in body.get("models", []):
        name = model.get("name") or model.get("model")
        if name:
            models.append(name)
    return tuple(models)


def _ollama_model_available(model_name: str) -> bool:
    return model_name in _ollama_list_models(_get_ollama_base_url())


def _resolve_provider_and_model(
    provider: str | None,
    model: str | None,
) -> tuple[str, str | None]:
    resolved_provider = provider or os.environ.get("CRG_EMBEDDING_PROVIDER")
    resolved_model = model or os.environ.get("CRG_EMBEDDING_MODEL")

    if resolved_provider:
        return resolved_provider, resolved_model

    if resolved_model:
        if _ollama_model_available(resolved_model):
            return "ollama", resolved_model
        return "local", resolved_model

    if _ollama_model_available(OLLAMA_DEFAULT_MODEL):
        return "ollama", OLLAMA_DEFAULT_MODEL

    return "local", None


def get_provider(
    provider: str | None = None,
    model: str | None = None,
    allow_cold_load: bool = True,
) -> EmbeddingProvider | None:
    """Get an embedding provider by name.

    Args:
        provider: Provider name. One of "local", "ollama", "google", "minimax",
                  or None for automatic selection.
                  Google requires GOOGLE_API_KEY env var and explicit opt-in.
                  MiniMax requires MINIMAX_API_KEY env var and explicit opt-in.
                  Cloud providers emit a one-time stderr warning before use
                  unless ``CRG_ACCEPT_CLOUD_EMBEDDINGS=1`` is set. See: #174
        model: Model name/path to use. For local provider this is any
               sentence-transformers compatible model. Falls back to
               CRG_EMBEDDING_MODEL env var, then prefers
               nomic-embed-text-v2-moe:latest via Ollama when available,
               otherwise all-MiniLM-L6-v2.
               For Google provider this is a Gemini model ID.
        allow_cold_load: When False, do not initialize the sentence-transformers
                         local provider. Query-time search uses this to avoid
                         expensive local model loads on a cold server process.
    """
    provider, model = _resolve_provider_and_model(provider, model)

    if provider == "minimax":
        api_key = os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError(
                "MINIMAX_API_KEY environment variable is required for "
                "the MiniMax embedding provider."
            )
        _warn_cloud_egress("minimax")
        return MiniMaxEmbeddingProvider(api_key=api_key)

    if provider == "google":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for "
                "the Google embedding provider."
            )
        _warn_cloud_egress("google")
        try:
            return GoogleEmbeddingProvider(
                api_key=api_key,
                **({"model": model} if model else {}),
            )
        except ImportError:
            return None

    if provider == "ollama":
        return OllamaEmbeddingProvider(model_name=model)

    if not allow_cold_load:
        return None

    try:
        return LocalEmbeddingProvider(model_name=model)
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _check_available() -> bool:
    """Check whether local embedding support is available."""
    return importlib.util.find_spec("sentence_transformers") is not None


# ---------------------------------------------------------------------------
# SQLite vector storage
# ---------------------------------------------------------------------------

_EMBEDDINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    qualified_name TEXT PRIMARY KEY,
    vector BLOB NOT NULL,
    text_hash TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'unknown'
);
"""


def _encode_vector(vec: list[float]) -> bytes:
    """Encode a float vector as a compact binary blob."""
    return struct.pack(f"{len(vec)}f", *vec)


def _decode_vector(blob: bytes) -> list[float]:
    """Decode a binary blob back to a float vector."""
    n = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f"{n}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _node_to_text(node: GraphNode) -> str:
    """Convert a node to a searchable text representation."""
    parts = [node.name]
    if node.kind != "File":
        parts.append(node.kind.lower())
    if node.parent_name:
        parts.append(f"in {node.parent_name}")
    if node.params:
        parts.append(node.params)
    if node.return_type:
        parts.append(f"returns {node.return_type}")
    if node.language:
        parts.append(node.language)
    return " ".join(parts)


class EmbeddingStore:
    """Manages vector embeddings for graph nodes in SQLite."""

    def __init__(
        self,
        db_path: str | Path,
        provider: str | None = None,
        model: str | None = None,
        allow_cold_load: bool = True,
    ) -> None:
        self.provider = get_provider(
            provider,
            model=model,
            allow_cold_load=allow_cold_load,
        )
        self.available = self.provider is not None
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(
            str(self.db_path), timeout=30, check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_EMBEDDINGS_SCHEMA)

        # Migration for existing DBs missing the provider column
        try:
            self._conn.execute("SELECT provider FROM embeddings LIMIT 1")
        except sqlite3.OperationalError:
            self._conn.execute(
                "ALTER TABLE embeddings ADD COLUMN provider "
                "TEXT NOT NULL DEFAULT 'unknown'"
            )

        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def embed_nodes(self, nodes: list[GraphNode], batch_size: int = 64) -> int:
        """Compute and store embeddings for a list of nodes."""
        if not self.provider:
            return 0

        # Filter to nodes that need embedding
        to_embed: list[tuple[GraphNode, str, str]] = []
        provider_name = self.provider.name

        for node in nodes:
            if node.kind == "File":
                continue
            text = _node_to_text(node)
            text_hash = hashlib.sha256(text.encode()).hexdigest()

            existing = self._conn.execute(
                "SELECT text_hash, provider FROM embeddings WHERE qualified_name = ?",
                (node.qualified_name,),
            ).fetchone()

            # Re-embed if text changed OR provider changed
            if (existing and existing["text_hash"] == text_hash
                    and existing["provider"] == provider_name):
                continue
            to_embed.append((node, text, text_hash))

        if not to_embed:
            return 0

        # Encode in batches
        texts = [t for _, t, _ in to_embed]
        vectors = self.provider.embed(texts)

        for (node, _text, text_hash), vec in zip(to_embed, vectors):
            blob = _encode_vector(vec)
            self._conn.execute(
                """INSERT OR REPLACE INTO embeddings (qualified_name, vector, text_hash, provider)
                   VALUES (?, ?, ?, ?)""",
                (node.qualified_name, blob, text_hash, provider_name),
            )

        self._conn.commit()
        return len(to_embed)

    def search(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        """Search for nodes by semantic similarity."""
        if not self.provider:
            return []

        provider_name = self.provider.name
        query_vec = self.provider.embed_query(query)

        # Process in chunks, only matching current provider
        scored: list[tuple[str, float]] = []
        cursor = self._conn.execute(
            "SELECT qualified_name, vector FROM embeddings WHERE provider = ?",
            (provider_name,),
        )
        chunk_size = 500
        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            for row in rows:
                vec = _decode_vector(row["vector"])
                sim = _cosine_similarity(query_vec, vec)
                scored.append((row["qualified_name"], sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def remove_node(self, qualified_name: str) -> None:
        self._conn.execute(
            "DELETE FROM embeddings WHERE qualified_name = ?", (qualified_name,)
        )
        self._conn.commit()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]


def embed_all_nodes(graph_store: GraphStore, embedding_store: EmbeddingStore) -> int:
    """Embed all non-file nodes in the graph."""
    if not embedding_store.available:
        return 0

    all_files = graph_store.get_all_files()
    all_nodes: list[GraphNode] = []
    for f in all_files:
        all_nodes.extend(graph_store.get_nodes_by_file(f))

    return embedding_store.embed_nodes(all_nodes)


def semantic_search(
    query: str,
    graph_store: GraphStore,
    embedding_store: EmbeddingStore,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search nodes using vector similarity, falling back to keyword search."""
    if embedding_store.available and embedding_store.count() > 0:
        results = embedding_store.search(query, limit=limit)
        output = []
        for qn, score in results:
            node = graph_store.get_node(qn)
            if node:
                d = node_to_dict(node)
                d["similarity_score"] = round(score, 4)
                output.append(d)
        return output

    # Fallback to keyword search
    nodes = graph_store.search_nodes(query, limit=limit)
    return [node_to_dict(n) for n in nodes]









