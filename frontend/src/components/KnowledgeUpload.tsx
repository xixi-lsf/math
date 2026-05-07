import { useState, useEffect, useRef, DragEvent, ChangeEvent } from 'react'
import { BookOpen, Upload, Trash2, CheckCircle, XCircle, Loader2 } from 'lucide-react'
import clsx from 'clsx'

interface DocMeta {
  doc_id: string
  filename: string
  chunks_count: number
  uploaded_at: string
  doc_type: 'knowledge' | 'problem'
}

interface UploadingFile {
  name: string
  doc_type: 'knowledge' | 'problem'
  status: 'uploading' | 'success' | 'error'
  error?: string
}

interface Props {
  onSelectionChange: (selectedKnowledgeIds: string[], selectedProblemIds: string[]) => void
}

const MAX_SIZE = 20 * 1024 * 1024

function UploadSection({
  label,
  hint,
  docType,
  docs,
  uploading,
  checkedIds,
  onToggle,
  onUpload,
  onDelete,
  onDismiss,
}: {
  label: string
  hint: string
  docType: 'knowledge' | 'problem'
  docs: DocMeta[]
  uploading: UploadingFile[]
  checkedIds: Set<string>
  onToggle: (id: string) => void
  onUpload: (file: File, docType: 'knowledge' | 'problem') => void
  onDelete: (doc: DocMeta) => void
  onDismiss: (name: string) => void
}) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const sectionUploading = uploading.filter(f => f.doc_type === docType)

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragging(false)
    Array.from(e.dataTransfer.files).forEach(f => onUpload(f, docType))
  }

  function handleInputChange(e: ChangeEvent<HTMLInputElement>) {
    Array.from(e.target.files ?? []).forEach(f => onUpload(f, docType))
    e.target.value = ''
  }

  return (
    <div>
      <p className="text-xs font-medium text-slate-600 mb-1.5">{label}</p>

      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        className={clsx(
          'border-2 border-dashed rounded-lg px-3 py-3 text-center cursor-pointer transition-colors',
          dragging ? 'border-indigo-400 bg-indigo-50' : 'border-slate-200 hover:border-indigo-300 hover:bg-slate-50'
        )}
      >
        <Upload size={14} className="mx-auto mb-1 text-slate-400" />
        <p className="text-xs text-slate-500">{hint}</p>
      </div>
      <input ref={inputRef} type="file" accept=".pdf,.docx,.txt" multiple className="hidden" onChange={handleInputChange} />

      {/* Upload status */}
      {sectionUploading.length > 0 && (
        <ul className="mt-1.5 space-y-1">
          {sectionUploading.map(f => (
            <li key={f.name} className="flex items-center gap-1.5 text-xs">
              {f.status === 'uploading' && <Loader2 size={12} className="text-indigo-500 animate-spin flex-shrink-0" />}
              {f.status === 'success'  && <CheckCircle size={12} className="text-green-500 flex-shrink-0" />}
              {f.status === 'error'    && <XCircle size={12} className="text-red-500 flex-shrink-0" />}
              <span className={clsx('flex-1 truncate', f.status === 'error' ? 'text-red-600' : 'text-slate-600')}>
                {f.name}{f.error && <span className="text-red-400 ml-1">— {f.error}</span>}
              </span>
              {f.status !== 'uploading' && (
                <button onClick={() => onDismiss(f.name)} className="text-slate-400 hover:text-slate-600 text-xs">×</button>
              )}
            </li>
          ))}
        </ul>
      )}

      {/* Doc list with checkboxes */}
      {docs.length > 0 && (
        <ul className="mt-1.5 space-y-1">
          {docs.map(doc => (
            <li key={doc.doc_id} className="flex items-center gap-2 text-xs bg-slate-50 rounded-lg px-2.5 py-1.5">
              <input
                type="checkbox"
                checked={checkedIds.has(doc.doc_id)}
                onChange={() => onToggle(doc.doc_id)}
                className="accent-indigo-600 flex-shrink-0"
              />
              <span className="flex-1 truncate text-slate-700">{doc.filename}</span>
              <span className="text-slate-400 flex-shrink-0">{doc.chunks_count}块</span>
              <button
                onClick={() => onDelete(doc)}
                className="text-slate-400 hover:text-red-500 transition-colors flex-shrink-0"
                title="删除"
              >
                <Trash2 size={12} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default function KnowledgeUpload({ onSelectionChange }: Props) {
  const [docs, setDocs] = useState<DocMeta[]>([])
  const [uploading, setUploading] = useState<UploadingFile[]>([])
  const [checkedIds, setCheckedIds] = useState<Set<string>>(new Set())

  useEffect(() => { fetchDocs() }, [])

  // Notify parent whenever checked set changes
  useEffect(() => {
    const knowledgeIds = docs
      .filter(d => d.doc_type === 'knowledge' && checkedIds.has(d.doc_id))
      .map(d => d.doc_id)
    const problemIds = docs
      .filter(d => d.doc_type === 'problem' && checkedIds.has(d.doc_id))
      .map(d => d.doc_id)
    onSelectionChange(knowledgeIds, problemIds)
  }, [checkedIds, docs])

  async function fetchDocs() {
    try {
      const res = await fetch('/api/v1/knowledge/documents')
      if (res.ok) setDocs(await res.json())
    } catch { /* ignore */ }
  }

  async function uploadFile(file: File, docType: 'knowledge' | 'problem') {
    if (file.size > MAX_SIZE) {
      setUploading(prev => [...prev, { name: file.name, doc_type: docType, status: 'error', error: '超过 20MB' }])
      return
    }
    setUploading(prev => [...prev, { name: file.name, doc_type: docType, status: 'uploading' }])
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('doc_type', docType)
      const res = await fetch('/api/v1/knowledge/upload', { method: 'POST', body: formData })
      const data = await res.json()
      if (!res.ok) {
        setUploading(prev => prev.map(f => f.name === file.name ? { ...f, status: 'error', error: data.detail ?? '上传失败' } : f))
        return
      }
      setUploading(prev => prev.map(f => f.name === file.name ? { ...f, status: 'success' } : f))
      await fetchDocs()
    } catch (e) {
      setUploading(prev => prev.map(f => f.name === file.name ? { ...f, status: 'error', error: String(e) } : f))
    }
  }

  function toggleCheck(id: string) {
    setCheckedIds(prev => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  async function deleteDoc(doc: DocMeta) {
    await fetch(`/api/v1/knowledge/documents/${doc.doc_id}?doc_type=${doc.doc_type}`, { method: 'DELETE' })
    setDocs(prev => prev.filter(d => d.doc_id !== doc.doc_id))
    setCheckedIds(prev => { const n = new Set(prev); n.delete(doc.doc_id); return n })
  }

  const knowledgeDocs = docs.filter(d => d.doc_type === 'knowledge')
  const problemDocs   = docs.filter(d => d.doc_type === 'problem')

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
      <h2 className="font-semibold text-slate-800 text-sm uppercase tracking-wide flex items-center gap-1.5">
        <BookOpen size={14} className="text-indigo-500" />
        知识库
      </h2>

      <UploadSection
        label="自建知识库"
        hint="上传教材、笔记等知识文档（.pdf / .docx / .txt）"
        docType="knowledge"
        docs={knowledgeDocs}
        uploading={uploading}
        checkedIds={checkedIds}
        onToggle={toggleCheck}
        onUpload={uploadFile}
        onDelete={deleteDoc}
        onDismiss={name => setUploading(prev => prev.filter(f => f.name !== name))}
      />

      <div className="border-t border-slate-100" />

      <UploadSection
        label="自建例题库"
        hint="上传真题、练习册等题目文档（.pdf / .docx / .txt）"
        docType="problem"
        docs={problemDocs}
        uploading={uploading}
        checkedIds={checkedIds}
        onToggle={toggleCheck}
        onUpload={uploadFile}
        onDelete={deleteDoc}
        onDismiss={name => setUploading(prev => prev.filter(f => f.name !== name))}
      />

      <p className="text-xs text-slate-400">勾选文档后，生成题目时将优先参考勾选内容</p>
    </div>
  )
}
