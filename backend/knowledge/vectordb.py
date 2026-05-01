"""
ChromaDB vector store wrapper for the knowledge base.
Supports builtin theorems, user-uploaded documents, and web-search results.

Embedding strategy (in priority order):
1. OpenAI-compatible embedding API (if EMBEDDING_BASE_URL + EMBEDDING_API_KEY set)
2. Sentence-transformers (if already downloaded)
3. Simple keyword-based fallback (no download needed)
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from models.knowledge import KnowledgeChunk

# ── Paths ────────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent
_BUILTIN_DIR = _BASE_DIR / "builtin"
_CHROMA_DIR = _BASE_DIR / "chroma_db"

# ── Embedding function ────────────────────────────────────────────────────────
#关键词嵌入
class _KeywordEmbeddingFunction:
    """
    Lightweight keyword-based embedding (no model download).
    Uses a fixed vocabulary of math terms to create sparse vectors.
    Good enough for small knowledge bases (<200 documents).
    """
    #维护一个固定的数学词汇表
    _VOCAB = [
        "椭圆", "双曲线", "抛物线", "极坐标", "焦点", "准线", "渐近线",
        "切线", "法线", "弦", "通径", "焦点弦", "离心率", "半长轴", "半短轴",
        "焦距", "顶点", "面积", "三角形", "斜率", "截距", "交点", "参数方程",
        "标准方程", "光学性质", "反射", "圆锥曲线", "统一方程", "极径",
        "ellipse", "hyperbola", "parabola", "polar", "focal", "tangent",
        "chord", "asymptote", "eccentricity", "directrix", "vertex",
    ]

    def name(self) -> str:
        return "keyword_embedding"

    def __call__(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            vec = [1.0 if kw in text else 0.0 for kw in self._VOCAB]
            # Normalize
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            result.append([v / norm for v in vec])
        return result

#嵌入函数
def _make_embed_fn():
    # Try sentence-transformers (already downloaded)
    try:
        import sentence_transformers  # noqa: F401
        # 查找缓存目录下是否存在以 "models--sentence-transformers" 开头的文件夹
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_dirs = list(cache_dir.glob("models--sentence-transformers*"))
        # 如果存在，说明模型已下载
        if model_dirs:
            # 返回 ChromaDB 内置的 SentenceTransformer 嵌入函数
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
    except Exception:
        pass
    # Fallback: keyword-based
    return _KeywordEmbeddingFunction()


_EMBED_FN = _make_embed_fn()


class KnowledgeStore:
    """Thin wrapper around ChromaDB for retrieval and ingestion."""

    #PersistentClient 将向量数据持久化到磁盘（chroma_db 目录），下次启动无需重新加载
    #创建了两个 ChromaDB 集合，这两个集合数据隔离
    #_theorems：存储内置知识和网络搜索结果
    #_user_docs：存储用户上传文档
    def __init__(self, persist_dir: Optional[str] = None):
        path = persist_dir or str(_CHROMA_DIR)
        self._client = chromadb.PersistentClient(path=path)
        self._theorems = self._client.get_or_create_collection(
            name="theorems", embedding_function=_EMBED_FN
        )
        self._user_docs = self._client.get_or_create_collection(
            name="user_documents", embedding_function=_EMBED_FN
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    #内置知识加载
    #self 指向 KnowledgeStore 对象
    def load_builtin(self) -> int:
        """增量加载内置知识库。内容无变化时直接跳过，有变化时重建 collection。"""
        # 第一步：收集所有 builtin chunks
        all_chunks: list[dict] = []
        for topic_dir in sorted(_BUILTIN_DIR.iterdir()):
            if not topic_dir.is_dir():
                continue
            for json_file in sorted(topic_dir.glob("*.json")):
                chunks: list[dict] = json.loads(json_file.read_text(encoding="utf-8"))
                all_chunks.extend(chunks)

        # 第二步：计算整体内容哈希
        content_hash = hashlib.md5(
            json.dumps(all_chunks, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

        # 第三步：检查已存储的哈希标记
        try:
            existing = self._theorems.get(ids=["__builtin_hash__"])
            if existing["ids"] and existing["documents"][0] == content_hash:
                print(f"[KnowledgeStore] builtin 内容未变化（hash={content_hash[:8]}），跳过重建")
                return 0
        except Exception:
            pass

        # 第四步：哈希不一致或不存在，重建 collection
        print(f"[KnowledgeStore] builtin 内容已变化，重建 theorems collection（{len(all_chunks)} 条）")
        try:
            self._client.delete_collection("theorems")
        except Exception:
            pass
        self._theorems = self._client.get_or_create_collection(
            name="theorems", embedding_function=_EMBED_FN
        )

        for chunk in all_chunks:
            self._theorems.add(
                ids=[chunk["id"]],
                documents=[chunk["content"]],
                metadatas=[{
                    "latex_formula": chunk.get("latex_formula", ""),
                    "topic": chunk.get("topic", ""),
                    "subtopic": chunk.get("subtopic", ""),
                    "difficulty_min": int(chunk.get("difficulty_range", [1, 5])[0]),
                    "difficulty_max": int(chunk.get("difficulty_range", [1, 5])[1]),
                    "source": "builtin",
                }],
            )

        # 写入哈希标记
        self._theorems.add(
            ids=["__builtin_hash__"],
            documents=[content_hash],
            metadatas=[{"source": "hash_marker", "topic": "__meta__",
                        "difficulty_min": 0, "difficulty_max": 0}],
        )
        return len(all_chunks)

    #添加用户文档
    def add_user_document(self, doc_id: str, text: str, metadata: dict) -> None:
        """Add a user-uploaded document chunk."""
        self._user_docs.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[{**metadata, "source": "user_upload"}],
        )

    #缓存网络结果
    def add_web_result(self, chunk: KnowledgeChunk) -> None:
        """Cache a web-search result into the theorems collection."""
        self._theorems.add(
            ids=[chunk.id],
            documents=[chunk.content],
            metadatas=[{
                "latex_formula": chunk.latex_formula,
                "topic": chunk.topic,
                "subtopic": chunk.subtopic,
                "difficulty_min": chunk.difficulty_range[0],
                "difficulty_max": chunk.difficulty_range[1],
                "source": "web_search",
            }],
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        topic: str,
        difficulty: int,
        n_results: int = 6,
    ) -> list[KnowledgeChunk]:
        """Semantic search filtered by topic and difficulty."""
        where = {
            "$and": [
                {"topic": {"$eq": topic}},
                {"difficulty_min": {"$lte": difficulty}},
                {"difficulty_max": {"$gte": difficulty}},
            ]
        }
        try:
            results = self._theorems.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
            )
        except Exception:
            # Fallback: no filter (e.g. collection empty)
            results = self._theorems.query(query_texts=[query], n_results=n_results)

        chunks: list[KnowledgeChunk] = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            chunks.append(KnowledgeChunk(
                id=doc_id,
                content=results["documents"][0][i],
                latex_formula=meta.get("latex_formula", ""),
                topic=meta.get("topic", topic),
                subtopic=meta.get("subtopic", ""),
                difficulty_range=(
                    meta.get("difficulty_min", 1),
                    meta.get("difficulty_max", 5),
                ),
                source=meta.get("source", "builtin"),
            ))
        return chunks

    def retrieve_user_docs(self, query: str, n_results: int = 4) -> list[KnowledgeChunk]:
        """Search user-uploaded documents."""
        try:
            results = self._user_docs.query(query_texts=[query], n_results=n_results)
        except Exception:
            return []
        chunks: list[KnowledgeChunk] = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            chunks.append(KnowledgeChunk(
                id=doc_id,
                content=results["documents"][0][i],
                latex_formula=meta.get("latex_formula", ""),
                topic=meta.get("topic", ""),
                subtopic=meta.get("subtopic", ""),
                source="user_upload",
            ))
        return chunks

    def list_user_docs(self) -> list[dict]:
        """List all user-uploaded document metadata."""
        result = self._user_docs.get()
        seen_sources: dict[str, dict] = {}
        for i, doc_id in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            src = meta.get("filename", doc_id)
            if src not in seen_sources:
                seen_sources[src] = {"id": doc_id, **meta}
        return list(seen_sources.values())

    def delete_user_doc(self, doc_id: str) -> None:
        self._user_docs.delete(ids=[doc_id])


# ── Singleton ─────────────────────────────────────────────────────────────────
_store: Optional[KnowledgeStore] = None


def get_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
        _store.load_builtin()
    return _store
