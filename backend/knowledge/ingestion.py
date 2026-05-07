"""
Document ingestion: parse PDF/TXT/MD files and add to user_documents collection.
实现了文档摄取功能，允许用户上传 PDF、DOCX、TXT/Markdown 文件，
将内容分块后存入向量数据库（ChromaDB）的用户文档集合中，供后续知识检索使用
"""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
from typing import Union

from knowledge.vectordb import get_store


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

#参数：文件路径，主题，可选 doc_id（用于后续按文档删除）
def ingest_file(file_path: Union[str, Path], topic: str = "general", doc_id: str | None = None) -> int:
    """
    Parse a file and add its chunks to the user_documents collection (legacy).
    Returns the number of chunks added.
    If doc_id is provided, it is stored in chunk metadata for later bulk deletion.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix in (".docx", ".doc"):
        text = _read_docx(path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    chunks = _chunk_text(text)
    store = get_store()
    added = 0
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        chunk_id = hashlib.md5(f"{path.name}_{idx}_{chunk[:50]}".encode()).hexdigest()
        meta: dict = {"filename": path.name, "topic": topic, "chunk_index": idx}
        if doc_id:
            meta["doc_id"] = doc_id
        store.add_user_document(
            doc_id=chunk_id,
            text=chunk,
            metadata=meta,
        )
        added += 1
    return added


def ingest_file_typed(
    file_path: Union[str, Path],
    doc_id: str,
    doc_type: str,  # "knowledge" | "problem"
    topic: str = "general",
) -> int:
    """
    Parse a file and route chunks to user_knowledge or user_problems collection.
    Returns the number of chunks added.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix in (".docx", ".doc"):
        text = _read_docx(path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    chunks = _chunk_text(text)
    store = get_store()
    added = 0
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        chunk_id = hashlib.md5(f"{doc_id}_{idx}_{chunk[:50]}".encode()).hexdigest()
        meta: dict = {
            "filename": path.name,
            "topic": topic,
            "chunk_index": idx,
            "doc_id": doc_id,
            "doc_type": doc_type,
        }
        if doc_type == "problem":
            store.add_user_problem_chunk(chunk_id, chunk, meta)
        else:
            store.add_user_knowledge_chunk(chunk_id, chunk, meta)
        added += 1
    return added
