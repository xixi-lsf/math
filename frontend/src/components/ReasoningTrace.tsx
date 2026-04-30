import { useEffect, useRef } from 'react'
import type { ReasoningStep } from '../types'
import { NODE_LABELS, NODE_COLORS } from '../types'
import { CheckCircle, Loader, AlertCircle, Zap, Cpu } from 'lucide-react'
import clsx from 'clsx'

interface Props {
  steps: ReasoningStep[]
  isStreaming: boolean
}

export default function ReasoningTrace({ steps, isStreaming }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [steps.length])

  if (steps.length === 0 && !isStreaming) return null

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-slate-600 uppercase tracking-wide flex items-center gap-1.5">
        <Cpu size={14} />
        推理过程
      </h3>

      <div className="space-y-1.5 max-h-72 overflow-y-auto pr-1">
        {steps.map((step, idx) => (
          <StepCard key={step.step_id} step={step} isLast={idx === steps.length - 1 && isStreaming} />
        ))}

        {isStreaming && steps.length === 0 && (
          <div className="flex items-center gap-2 text-sm text-slate-500 py-2">
            <Loader size={14} className="animate-spin" />
            Agent 启动中...
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  )
}

function StepCard({ step, isLast }: { step: ReasoningStep; isLast: boolean }) {
  const colorClass = NODE_COLORS[step.node_name] ?? 'bg-slate-100 text-slate-800 border-slate-200'
  const label = NODE_LABELS[step.node_name] ?? step.node_name

  return (
    <div className={clsx('step-enter rounded-lg border p-2.5 text-xs', colorClass)}>
      <div className="flex items-start gap-2">
        <div className="mt-0.5 flex-shrink-0">
          {isLast ? (
            <Loader size={13} className="animate-spin" />
          ) : (
            <CheckCircle size={13} />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="font-semibold">{label}</span>
            {step.drawing_path && (
              <span className={clsx(
                'px-1.5 py-0.5 rounded text-[10px] font-medium',
                step.drawing_path === 'fast'
                  ? 'bg-green-200 text-green-800'
                  : 'bg-orange-200 text-orange-800'
              )}>
                {step.drawing_path === 'fast' ? '⚡ 快速路径' : '🔧 代码生成'}
              </span>
            )}
          </div>
          <p className="mt-0.5 text-[11px] opacity-80 leading-relaxed">{step.action}</p>
          {step.tool_called && (
            <p className="mt-0.5 font-mono text-[10px] opacity-60">
              🔧 {step.tool_called}
            </p>
          )}
          {step.tool_output_summary && (
            <p className="mt-0.5 text-[10px] opacity-60 truncate">
              → {step.tool_output_summary}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
