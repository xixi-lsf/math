from typing import Annotated, Optional
from typing_extensions import TypedDict
from models.problem import ProblemParams, Problem, ReasoningStep, ValidationResult
from models.knowledge import KnowledgeChunk


class AgentState(TypedDict):
    # ── Inputs ──────────────────────────────────────────────────────────────
    topic: str
    difficulty: int
    subtopics: list[str]
    llm_config: dict          # {base_url, api_key, model}

    # ── Intermediate ─────────────────────────────────────────────────────────
    retrieved_knowledge: list[KnowledgeChunk]
    latex_problem: Optional[str]
    params: Optional[ProblemParams]
    validation_result: Optional[ValidationResult]
    image_base64: Optional[str]
    drawing_path: Optional[str]   # "fast" | "slow"
    drawing_code: Optional[str]   # LLM-generated code (slow path)
    drawing_error: Optional[str]  # last execution error (for retry)
    retry_count: int
    drawing_retry_count: int
    reasoning_trace: list[ReasoningStep]
    step_counter: int

    # ── Output ───────────────────────────────────────────────────────────────
    final_problem: Optional[Problem]
    solution_latex: Optional[str]
    error_message: Optional[str]
