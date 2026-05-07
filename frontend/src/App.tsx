import { useState } from 'react'
import { useSSEStream } from './hooks/useSSEStream'
import ProblemForm from './components/ProblemForm'
import ReasoningTrace from './components/ReasoningTrace'
import ProblemDisplay from './components/ProblemDisplay'
import LLMConfigPanel from './components/LLMConfigPanel'
import KnowledgeUpload from './components/KnowledgeUpload'
import { loadConfig } from './config/llmConfig'
import type { LLMConfig } from './config/llmConfig'
import type { Topic } from './types'
import { AlertCircle, FlaskConical } from 'lucide-react'

export default function App() {
  const [llmConfig, setLlmConfig] = useState<LLMConfig>(loadConfig)
  const [solution, setSolution] = useState<string | null>(null)
  const [isSolving, setIsSolving] = useState(false)
  const [selectedKnowledgeIds, setSelectedKnowledgeIds] = useState<string[]>([])
  const [selectedProblemIds, setSelectedProblemIds] = useState<string[]>([])

  const { steps, problem, isStreaming, error, start, reset } = useSSEStream()

  const handleGenerate = (topic: Topic, difficulty: number, subtopics: string[], config: LLMConfig) => {
    setSolution(null)
    reset()
    start(topic, difficulty, subtopics, config, selectedKnowledgeIds, selectedProblemIds)
  }

  const handleRequestSolution = async () => {
    if (!problem || isSolving) return
    setIsSolving(true)
    try {
      const res = await fetch('/api/v1/problems/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          latex_problem: problem.latex_problem,
          params: problem.params,
          llm_config: llmConfig,
        }),
      })
      const data = await res.json()
      setSolution(data.solution ?? data.solution_latex ?? '')
    } catch {
      setSolution('解题步骤生成失败，请重试。')
    } finally {
      setIsSolving(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-indigo-50/30">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-40 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FlaskConical size={20} className="text-indigo-600" />
            <span className="font-bold text-slate-900">解析几何题目生成系统</span>
            <span className="text-xs text-slate-400 hidden sm:block">— Analytical Geometry Problem Generator</span>
          </div>
          <LLMConfigPanel config={llmConfig} onChange={setLlmConfig} />
        </div>
      </header>

      {/* Main layout */}
      <main className="max-w-6xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">

          {/* Left panel: form + reasoning trace */}
          <div className="space-y-5">
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <h2 className="font-semibold text-slate-800 mb-4 text-sm uppercase tracking-wide">生成设置</h2>
              <ProblemForm
                onGenerate={handleGenerate}
                isLoading={isStreaming}
                llmConfig={llmConfig}
              />
            </div>

            {/* Reasoning trace */}
            {(steps.length > 0 || isStreaming) && (
              <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
                <ReasoningTrace steps={steps} isStreaming={isStreaming} />
              </div>
            )}

            <KnowledgeUpload
              onSelectionChange={(kIds, pIds) => {
                setSelectedKnowledgeIds(kIds)
                setSelectedProblemIds(pIds)
              }}
            />
          </div>

          {/* Right panel: problem display */}
          <div>
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-2 text-sm text-red-700">
                <AlertCircle size={16} className="mt-0.5 flex-shrink-0" />
                <div>
                  <p className="font-medium">生成失败</p>
                  <p className="mt-0.5 text-red-600">{error}</p>
                </div>
              </div>
            )}

            {!problem && !isStreaming && !error && (
              <div className="flex flex-col items-center justify-center h-80 text-slate-400 space-y-3">
                <FlaskConical size={48} className="opacity-20" />
                <p className="text-sm">选择知识点和难度，点击「生成题目」</p>
                <p className="text-xs opacity-70">Agent 将实时展示推理过程</p>
              </div>
            )}

            {isStreaming && !problem && (
              <div className="flex flex-col items-center justify-center h-80 text-slate-400 space-y-3">
                <div className="w-10 h-10 border-3 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                <p className="text-sm">Agent 正在生成题目...</p>
              </div>
            )}

            {problem && (
              <ProblemDisplay
                problem={problem}
                onRequestSolution={handleRequestSolution}
                solution={solution}
                isSolving={isSolving}
              />
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
