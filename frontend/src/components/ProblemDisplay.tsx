import { useState } from 'react'
import { InlineMath, BlockMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import type { ProblemResult } from '../types'
import { BookOpen, ChevronDown, ChevronUp, Zap, Wrench } from 'lucide-react'
import clsx from 'clsx'

interface Props {
  problem: ProblemResult
  onRequestSolution: () => void
  solution: string | null
  isSolving: boolean
}

export default function ProblemDisplay({ problem, onRequestSolution, solution, isSolving }: Props) {
  const [showSolution, setShowSolution] = useState(false)

  return (
    <div className="space-y-4">
      {/* Problem statement */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold text-slate-800 flex items-center gap-1.5">
            <BookOpen size={16} />
            题目
          </h2>
          <div className="flex items-center gap-1.5">
            {problem.is_fallback && (
              <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-slate-100 text-slate-700">
                📚 来自题库
              </span>
            )}
            {problem.drawing_path && (
              <span className={clsx(
                'text-xs px-2 py-0.5 rounded-full font-medium',
                problem.drawing_path === 'fast'
                  ? 'bg-green-100 text-green-700'
                  : 'bg-orange-100 text-orange-700'
              )}>
                {problem.drawing_path === 'fast' ? <><Zap size={10} className="inline mr-0.5" />快速绘图</> : <><Wrench size={10} className="inline mr-0.5" />代码绘图</>}
              </span>
            )}
            <span className="text-xs text-slate-400 font-mono">#{problem.problem_id}</span>
          </div>
        </div>

        <div className="prose prose-sm max-w-none text-slate-800 leading-relaxed">
          <LatexRenderer text={problem.latex_problem} />
        </div>
      </div>

      {/* Figure */}
      {problem.image_base64 && (
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
          <h3 className="text-sm font-semibold text-slate-600 mb-3">配图</h3>
          <div className="flex justify-center">
            <img
              src={`data:image/png;base64,${problem.image_base64}`}
              alt="解析几何配图"
              className="max-w-full rounded-lg"
              style={{ maxHeight: '420px' }}
            />
          </div>
        </div>
      )}

      {/* Solution panel */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <button
          onClick={() => {
            if (!solution && !isSolving) onRequestSolution()
            setShowSolution(v => !v)
          }}
          className="w-full flex items-center justify-between px-5 py-3.5 text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors"
        >
          <span className="flex items-center gap-1.5">
            <BookOpen size={15} />
            {isSolving ? '解题步骤生成中...' : '查看解题步骤'}
          </span>
          {showSolution ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
        </button>

        {showSolution && solution && (
          <div className="px-5 pb-5 border-t border-slate-100">
            <div className="mt-4 prose prose-sm max-w-none text-slate-800 leading-relaxed">
              <LatexRenderer text={solution} />
            </div>
          </div>
        )}

        {showSolution && isSolving && (
          <div className="px-5 pb-4 border-t border-slate-100">
            <div className="mt-3 flex items-center gap-2 text-sm text-slate-500">
              <div className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
              正在生成解题步骤...
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── LaTeX renderer ────────────────────────────────────────────────────────────

function LatexRenderer({ text }: { text: string }) {
  // Split text into segments: display math ($$...$$), inline math ($...$), and plain text
  const segments = parseLatex(text)
  return (
    <>
      {segments.map((seg, i) => {
        if (seg.type === 'display') {
          return (
            <div key={i} className="my-2">
              <BlockMath math={seg.content} />
            </div>
          )
        }
        if (seg.type === 'inline') {
          return <InlineMath key={i} math={seg.content} />
        }
        return <span key={i}>{seg.content}</span>
      })}
    </>
  )
}

interface Segment {
  type: 'text' | 'inline' | 'display'
  content: string
}

function parseLatex(text: string): Segment[] {
  const segments: Segment[] = []
  let remaining = text

  // Normalize \[...\] → $$...$$ and \(...\) → $...$
  remaining = remaining.replace(/\\\[([\s\S]*?)\\\]/g, (_m, c) => `$$${c}$$`)
  remaining = remaining.replace(/\\\(([\s\S]*?)\\\)/g, (_m, c) => `$${c}$`)

  while (remaining.length > 0) {
    // Display math: $$...$$
    const displayMatch = remaining.match(/\$\$([\s\S]*?)\$\$/)
    // Inline math: $...$
    const inlineMatch = remaining.match(/\$((?:[^$\\]|\\[\s\S])*?)\$/)

    const displayIdx = displayMatch?.index ?? Infinity
    const inlineIdx = inlineMatch?.index ?? Infinity

    if (displayIdx === Infinity && inlineIdx === Infinity) {
      segments.push({ type: 'text', content: remaining })
      break
    }

    if (displayIdx <= inlineIdx) {
      if (displayIdx > 0) {
        segments.push({ type: 'text', content: remaining.slice(0, displayIdx) })
      }
      segments.push({ type: 'display', content: displayMatch![1] })
      remaining = remaining.slice(displayIdx + displayMatch![0].length)
    } else {
      if (inlineIdx > 0) {
        segments.push({ type: 'text', content: remaining.slice(0, inlineIdx) })
      }
      segments.push({ type: 'inline', content: inlineMatch![1] })
      remaining = remaining.slice(inlineIdx + inlineMatch![0].length)
    }
  }

  return segments
}
