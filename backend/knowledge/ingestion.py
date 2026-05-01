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

#参数：文件路径，主题
def ingest_file(file_path: Union[str, Path], topic: str = "general") -> int:
    """
    Parse a file and add its chunks to the user_documents collection.
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
    #store提供两个相关方法：
    # add_user_document()将文本块插入名为 "user_documents" 的集合

    store = get_store()
    added = 0
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        chunk_id = hashlib.md5(f"{path.name}_{idx}_{chunk[:50]}".encode()).hexdigest()
        store.add_user_document(
            doc_id=chunk_id,
            text=chunk,
            metadata={"filename": path.name, "topic": topic, "chunk_index": idx},
        )
        added += 1
    return added
