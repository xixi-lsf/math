"""
FastAPI routes for knowledge base management.
"""
from __future__ import annotations
import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from knowledge.vectordb import get_store
from knowledge.ingestion import ingest_file_typed

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])

_KNOWLEDGE_META = Path(__file__).parent.parent / "knowledge" / "user_knowledge_meta.json"
_PROBLEMS_META  = Path(__file__).parent.parent / "knowledge" / "user_problems_meta.json"

_ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt"}
_MAX_BYTES = 20 * 1024 * 1024


# ── Metadata helpers ──────────────────────────────────────────────────────────

def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, data: list[dict]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_meta(path: Path, doc_id: str, filename: str, chunks_count: int) -> None:
    data = _load(path)
    data.append({
        "doc_id": doc_id,
        "filename": filename,
        "chunks_count": chunks_count,
        "uploaded_at": datetime.now().isoformat(),
    })
    _save(path, data)


def _remove_meta(path: Path, doc_id: str) -> None:
    _save(path, [d for d in _load(path) if d["doc_id"] != doc_id])


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form(...),  # "knowledge" | "problem"
):
    """Upload a document to user_knowledge or user_problems collection."""
    if doc_type not in ("knowledge", "problem"):
        raise HTTPException(400, detail="doc_type 必须是 knowledge 或 problem")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(400, detail=f"不支持的文件类型：{suffix}，支持 .pdf / .docx / .txt")

    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(400, detail="文件不能超过 20MB")

    tmp_path = Path(tempfile.mktemp(suffix=suffix))
    tmp_path.write_bytes(content)

    try:
        doc_id = str(uuid.uuid4())[:8]
        chunks_added = ingest_file_typed(str(tmp_path), doc_id=doc_id, doc_type=doc_type)
        meta_path = _KNOWLEDGE_META if doc_type == "knowledge" else _PROBLEMS_META
        _append_meta(meta_path, doc_id, file.filename or tmp_path.name, chunks_added)
        return {"doc_id": doc_id, "filename": file.filename, "chunks_added": chunks_added, "doc_type": doc_type}
    finally:
        tmp_path.unlink(missing_ok=True)


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("/documents")
async def list_documents():
    """Return all uploaded documents with their doc_type."""
    knowledge = [{"doc_type": "knowledge", **d} for d in _load(_KNOWLEDGE_META)]
    problems  = [{"doc_type": "problem",   **d} for d in _load(_PROBLEMS_META)]
    return knowledge + problems


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, doc_type: str = "knowledge"):
    """Delete a document and all its chunks from the appropriate collection."""
    store = get_store()
    if doc_type == "problem":
        store.delete_user_problem_doc(doc_id)
        _remove_meta(_PROBLEMS_META, doc_id)
    else:
        store.delete_user_knowledge_doc(doc_id)
        _remove_meta(_KNOWLEDGE_META, doc_id)
    return {"deleted": doc_id, "doc_type": doc_type}


# ── Legacy / utility ──────────────────────────────────────────────────────────

@router.post("/rebuild-index")
async def rebuild_index():
    """Reload all builtin knowledge into ChromaDB."""
    store = get_store()
    count = store.load_builtin()
    return {"message": f"重建索引完成，新增 {count} 条内置知识"}
