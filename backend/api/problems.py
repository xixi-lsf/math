"""
FastAPI routes for problem generation (SSE streaming) and on-demand solution.
"""
from __future__ import annotations
import asyncio
import json
import queue as _queue
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.graph import get_graph
from agent.nodes.solution_generation import solution_generation_node
from models.problem import ProblemParams

router = APIRouter(prefix="/api/v1/problems", tags=["problems"])

_problem_store: dict[str, object] = {}


class GenerateRequest(BaseModel):
    topic: str  # "ellipse" | "hyperbola" | "parabola" | "polar"
    difficulty: int = 3
    subtopics: list[str] = []
    llm_config: dict = {}
    selected_knowledge_ids: list[str] = []
    selected_problem_ids: list[str] = []


class SolveRequest(BaseModel):
    latex_problem: str
    params: ProblemParams | None = None
    llm_config: dict = {}


# ── SSE streaming generation ──────────────────────────────────────────────────

@router.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """
    Stream the agent's reasoning steps via SSE, then emit the final problem.
    """
    step_queue: _queue.Queue = _queue.Queue()

    initial_state = {
        "topic": req.topic,
        "difficulty": req.difficulty,
        "subtopics": req.subtopics,
        "llm_config": req.llm_config,
        "selected_knowledge_ids": req.selected_knowledge_ids,
        "selected_problem_ids": req.selected_problem_ids,
        "retrieved_knowledge": [],
        "latex_problem": None,
        "params": None,
        "validation_result": None,
        "solution": None,
        "image_base64": None,
        "drawing_path": None,
        "drawing_code": None,
        "drawing_error": None,
        "generation_retry": 0,
        "drawing_retry_count": 0,
        "reasoning_trace": [],
        "step_counter": 0,
        "step_queue": step_queue,
        "final_problem": None,
        "solution_latex": None,
        "error_message": None,
        "is_fallback": False,
    }

    async def simple_generator() -> AsyncGenerator[str, None]:
        graph = get_graph()
        loop = asyncio.get_event_loop()

        result_holder = {}

        def run_sync():
            try:
                final = graph.invoke(dict(initial_state))
                result_holder["final"] = final
            except Exception:
                import traceback
                result_holder["error"] = traceback.format_exc()
            finally:
                # Sentinel: signal that the graph has finished
                step_queue.put_nowait(None)

        task = loop.run_in_executor(None, run_sync)

        # Consume steps from queue in real-time until sentinel received
        while True:
            try:
                step = step_queue.get_nowait()
            except _queue.Empty:
                if task.done():
                    # Drain any remaining items after task completes
                    try:
                        step = step_queue.get_nowait()
                    except _queue.Empty:
                        break
                else:
                    await asyncio.sleep(0.05)
                    continue

            if step is None:
                # Sentinel received — graph is done
                break

            yield _sse_event("reasoning_step", {
                "step_id": step.step_id,
                "node_name": step.node_name,
                "action": step.action,
                "tool_called": step.tool_called,
                "tool_input_summary": step.tool_input_summary,
                "tool_output_summary": step.tool_output_summary,
                "drawing_path": step.drawing_path,
            })

        await task

        if "error" in result_holder:
            yield _sse_event("error", {"message": result_holder["error"]})
            return

        final = result_holder.get("final", {})
        problem = final.get("final_problem")
        if problem:
            _problem_store[problem.problem_id] = problem
            yield _sse_event("problem_ready", {
                "problem_id": problem.problem_id,
                "latex_problem": problem.latex_problem,
                "image_base64": problem.image_base64,
                "drawing_path": final.get("drawing_path"),
                "params": problem.params.model_dump() if problem.params else {},
                "is_fallback": problem.is_fallback,
            })
        else:
            yield _sse_event("error", {"message": "题目生成失败，请重试"})

    return StreamingResponse(
        simple_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── On-demand solution ────────────────────────────────────────────────────────

@router.post("/solve")
async def solve_problem_from_payload(req: SolveRequest):
    state = {
        "topic": req.params.topic if req.params else "ellipse",
        "difficulty": req.params.difficulty if req.params else 3,
        "subtopics": [],
        "llm_config": req.llm_config or {},
        "latex_problem": req.latex_problem,
        "params": req.params,
        "reasoning_trace": [],
        "step_counter": 0,
    }

    result = solution_generation_node(state)
    solution = result.get("solution_latex") or result.get("solution") or ""
    return {"solution": solution}


@router.get("/{problem_id}/solve")
async def solve_problem(problem_id: str):
    problem = _problem_store.get(problem_id)
    if not problem:
        raise HTTPException(404)
    if getattr(problem, "solution", None) is not None:
        return {"solution": problem.solution}

    # Fallback problems already have a solution from the problem bank — skip LLM
    if getattr(problem, "is_fallback", False):
        solution = getattr(problem, "solution_latex", "") or ""
        return {"solution": solution}

    state = {
        "topic": problem.params.topic if problem.params else "ellipse",
        "difficulty": problem.params.difficulty if problem.params else 3,
        "subtopics": [],
        "llm_config": getattr(problem, "generation_config", {}) or {},
        "latex_problem": problem.latex_problem,
        "params": problem.params,
        "reasoning_trace": [],
        "step_counter": 0,
    }

    result = solution_generation_node(state)
    solution = result.get("solution_latex", "")
    problem.solution = solution
    _problem_store[problem_id] = problem
    return {"solution": solution}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sse_event(event_type: str, data: dict) -> str:
    payload = {'type': event_type, **data}
    # 清理 image_base64 里可能存在的换行符，防止 SSE 帧被截断
    if 'image_base64' in payload and payload['image_base64']:
        payload['image_base64'] = payload['image_base64'].replace('\n', '').replace('\r', '')
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
