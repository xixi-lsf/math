import { useState, useCallback, useRef } from 'react'
import type { ReasoningStep, ProblemResult } from '../types'
import type { LLMConfig } from '../config/llmConfig'

interface UseSSEStreamResult {
  steps: ReasoningStep[]
  problem: ProblemResult | null
  isStreaming: boolean
  error: string | null
  start: (topic: string, difficulty: number, subtopics: string[], config: LLMConfig) => void
  reset: () => void
}

export function useSSEStream(): UseSSEStreamResult {
  const [steps, setSteps] = useState<ReasoningStep[]>([])
  const [problem, setProblem] = useState<ProblemResult | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setSteps([])
    setProblem(null)
    setError(null)
    setIsStreaming(false)
  }, [])

  const start = useCallback(
    async (topic: string, difficulty: number, subtopics: string[], config: LLMConfig) => {
      reset()
      setIsStreaming(true)

      const controller = new AbortController()
      abortRef.current = controller

      try {
        const response = await fetch('/api/v1/problems/generate/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ topic, difficulty, subtopics, llm_config: config }),
          signal: controller.signal,
        })

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }

        const reader = response.body!.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''

          let dataAccum = ''
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              dataAccum = line.slice(6)
            } else if (line === '') {
              // 空行代表一个 SSE 事件结束
              if (dataAccum) {
                try {
                  const event = JSON.parse(dataAccum.trim())
                  if (event.type === 'reasoning_step') {
                    setSteps(prev => [...prev, event as ReasoningStep])
                  } else if (event.type === 'problem_ready') {
                    setProblem(event as ProblemResult)
                    setIsStreaming(false)
                  } else if (event.type === 'error') {
                    setError(event.message ?? '未知错误')
                    setIsStreaming(false)
                  }
                } catch {}
                dataAccum = ''
              }
            } else if (dataAccum) {
              // 同一个 SSE 事件的续行，拼接起来
              dataAccum += line
            }
          }
        }
      } catch (e: unknown) {
        if ((e as Error).name !== 'AbortError') {
          setError((e as Error).message ?? '连接失败')
        }
      } finally {
        setIsStreaming(false)
      }
    },
    [reset]
  )

  return { steps, problem, isStreaming, error, start, reset }
}
