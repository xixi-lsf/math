// Shared types for the frontend
export interface ReasoningStep {
  step_id: number
  node_name: string
  action: string
  tool_called?: string
  tool_input_summary?: string
  tool_output_summary?: string
  drawing_path?: 'fast' | 'slow'
}

export interface ProblemResult {
  problem_id: string
  latex_problem: string
  image_base64: string
  drawing_path?: 'fast' | 'slow'
  params: Record<string, unknown>
}

export type Topic = 'ellipse' | 'hyperbola' | 'parabola' | 'polar'

export const TOPIC_LABELS: Record<Topic, string> = {
  ellipse: '椭圆',
  hyperbola: '双曲线',
  parabola: '抛物线',
  polar: '极坐标',
}

export const NODE_LABELS: Record<string, string> = {
  knowledge_retrieval: '知识检索',
  problem_generation: '题干生成',
  param_extraction: '参数提取',
  validation: '数学验证',
  drawing: '配图生成',
  solution_generation: '解题步骤',
  finalize: '完成',
}

export const NODE_COLORS: Record<string, string> = {
  knowledge_retrieval: 'bg-blue-100 text-blue-800 border-blue-200',
  problem_generation: 'bg-purple-100 text-purple-800 border-purple-200',
  param_extraction: 'bg-indigo-100 text-indigo-800 border-indigo-200',
  validation: 'bg-amber-100 text-amber-800 border-amber-200',
  drawing: 'bg-green-100 text-green-800 border-green-200',
  solution_generation: 'bg-teal-100 text-teal-800 border-teal-200',
  finalize: 'bg-slate-100 text-slate-800 border-slate-200',
}
