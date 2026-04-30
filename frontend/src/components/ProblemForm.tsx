import { useState } from 'react'
import type { Topic, LLMConfig } from '../types'
import { TOPIC_LABELS } from '../types'
import { Zap } from 'lucide-react'
import clsx from 'clsx'

interface Props {
  onGenerate: (topic: Topic, difficulty: number, subtopics: string[], config: LLMConfig) => void
  isLoading: boolean
  llmConfig: LLMConfig
}

const DIFFICULTY_LABELS = ['', '基础', '简单', '中等', '较难', '竞赛']

export default function ProblemForm({ onGenerate, isLoading, llmConfig }: Props) {
  const [topic, setTopic] = useState<Topic>('ellipse')
  const [difficulty, setDifficulty] = useState(3)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onGenerate(topic, difficulty, [], llmConfig)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      {/* Topic selection */}
      <div>
        <label className="block text-sm font-medium text-slate-700 mb-2">知识点</label>
        <div className="grid grid-cols-2 gap-2">
          {(Object.keys(TOPIC_LABELS) as Topic[]).map(t => (
            <button
              key={t}
              type="button"
              onClick={() => setTopic(t)}
              className={clsx(
                'py-2.5 px-3 rounded-lg border text-sm font-medium transition-all',
                topic === t
                  ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                  : 'bg-white text-slate-700 border-slate-200 hover:border-indigo-300 hover:bg-indigo-50'
              )}
            >
              {TOPIC_LABELS[t]}
            </button>
          ))}
        </div>
      </div>

      {/* Difficulty slider */}
      <div>
        <label className="block text-sm font-medium text-slate-700 mb-2">
          难度：
          <span className="ml-1 text-indigo-600 font-semibold">
            {difficulty}/5 — {DIFFICULTY_LABELS[difficulty]}
          </span>
        </label>
        <input
          type="range"
          min={1}
          max={5}
          value={difficulty}
          onChange={e => setDifficulty(Number(e.target.value))}
          className="w-full accent-indigo-600"
        />
        <div className="flex justify-between text-xs text-slate-400 mt-1">
          {DIFFICULTY_LABELS.slice(1).map((l, i) => (
            <span key={i}>{l}</span>
          ))}
        </div>
      </div>

      <button
        type="submit"
        disabled={isLoading}
        className={clsx(
          'w-full flex items-center justify-center gap-2 py-3 px-4 rounded-lg font-semibold text-sm transition-all',
          isLoading
            ? 'bg-indigo-400 text-white cursor-not-allowed'
            : 'bg-indigo-600 text-white hover:bg-indigo-700 active:scale-[0.98] shadow-sm'
        )}
      >
        <Zap size={16} />
        {isLoading ? '生成中...' : '生成题目'}
      </button>
    </form>
  )
}
