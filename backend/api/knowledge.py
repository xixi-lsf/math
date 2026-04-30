"""
FastAPI routes for knowledge base management.
"""
from __future__ import annotations
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from knowledge.vectordb import get_store
from knowledge.ingestion import ingest_file

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    topic: str = Form(default="general"),
):
    """Upload a PDF/TXT/MD/DOCX file to the user knowledge base."""
    allowed = {".pdf", ".txt", ".md", ".docx"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"不支持的文件类型 {suffix}，支持：{allowed}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        count = ingest_file(tmp_path, topic=topic)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"message": f"成功导入 {count} 个知识片段", "filename": file.filename, "chunks": count}


@router.get("/list")
async def list_documents():
    """List all user-uploaded documents."""
    store = get_store()
    docs = store.list_user_docs()
    return {"documents": docs}


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a user-uploaded document chunk."""
    store = get_store()
    store.delete_user_doc(doc_id)
    return {"message": f"已删除 {doc_id}"}


@router.post("/rebuild-index")
async def rebuild_index():
    """Reload all builtin knowledge into ChromaDB."""
    store = get_store()
    count = store.load_builtin()
    return {"message": f"重建索引完成，新增 {count} 条内置知识"}
