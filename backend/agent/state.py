from typing import Annotated, Optional, Any
from typing_extensions import TypedDict
from models.problem import ProblemParams, Problem, ReasoningStep, ValidationResult
from models.knowledge import KnowledgeChunk

"""
定义工作流中所有节点共享的状态结构（AgentState），包括输入、中间变量、输出。
描述“数据长什么样”
Optional 表示字段可能为 None
"""
class AgentState(TypedDict):
    # ── Inputs:题目主题，难度等级（1-5）,子主题列表，LLM配置────────────────────────
    topic: str
    difficulty: int
    subtopics: list[str]
    llm_config: dict          # {base_url, api_key, model}

    # ── User document selections (empty list = don't use user docs) ──
    selected_knowledge_ids: list[str]  # doc_ids from user_knowledge collection
    selected_problem_ids: list[str]    # doc_ids from user_problems collection

    # ── Intermediate（中间字段）──
    retrieved_knowledge: list[KnowledgeChunk]#知识检索节点找回的知识片段
    latex_problem: Optional[str]#生成的题目（LaTeX 格式）
    params: Optional[ProblemParams]#从题目中抽取的参数
    validation_result: Optional[ValidationResult]#solve_and_validate 节点输出的结果
    solution: Optional[str]#求解成功时存入的解题过程，供 finalize 使用
    image_base64: Optional[str]#题目配图的 base64 编码
    drawing_path: Optional[str]   # 绘图策略：fast预设模板，slowLLM生成代码
    drawing_code: Optional[str]   # LLM 生成的绘图代码
    drawing_error: Optional[str]  # 绘图执行时的错误信息，用于重试判断
    generation_retry: int #题目生成/求解失败后的重试次数
    drawing_retry_count: int #绘图执行失败后的重试次数
    reasoning_trace: list[ReasoningStep] #记录每一步 LLM 调用的推理过程
    step_counter: int #步骤计数器
    step_queue: Optional[Any]  # queue.Queue，用于实时推送推理步骤到 SSE 流

    # ── Output（输出字段）：───────────────────────────────────────────────────────────────
    final_problem: Optional[Problem]#最终生成的完整题目对象（文本+图片）
    solution_latex: Optional[str]#解题步骤的 LaTeX 表示
    error_message: Optional[str]#整个流程失败时的错误信息
    is_fallback: Optional[bool]#是否为题库保底题
