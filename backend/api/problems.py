"""
FastAPI routes for problem generation (SSE streaming) and on-demand solution.
"""
from __future__ import annotations
import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.graph import get_graph
from agent.nodes.solution_generation import solution_generation_node

router = APIRouter(prefix="/api/v1/problems", tags=["problems"])


class GenerateRequest(BaseModel):
    topic: str  # "ellipse" | "hyperbola" | "parabola" | "polar"
    difficulty: int = 3
    subtopics: list[str] = []
    llm_config: dict = {}


class SolveRequest(BaseModel):
    latex_problem: str
    params: dict
    llm_config: dict = {}


# ── SSE streaming generation ──────────────────────────────────────────────────

@router.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """
    Stream the agent's reasoning steps via SSE, then emit the final problem.
    """
    initial_state = {
        "topic": req.topic,
        "difficulty": req.difficulty,
        "subtopics": req.subtopics,
        "llm_config": req.llm_config,
        "retrieved_knowledge": [],
        "latex_problem": None,
        "params": None,
        "validation_result": None,
        "image_base64": None,
        "drawing_path": None,
        "drawing_code": None,
        "drawing_error": None,
        "retry_count": 0,
        "drawing_retry_count": 0,
        "reasoning_trace": [],
        "step_counter": 0,
        "final_problem": None,
        "solution_latex": None,
        "error_message": None,
    }

    async def event_generator() -> AsyncGenerator[str, None]:
        graph = get_graph()
        last_trace_len = 0

        try:
            # Run graph in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            final_state = await loop.run_in_executor(
                None, lambda: _run_graph_with_streaming(graph, initial_state, event_queue)
            )
        except Exception as e:
            yield _sse_event("error", {"message": str(e)})
            return

        # Drain any remaining events
        while not event_queue.empty():
            event = event_queue.get_nowait()
            yield event

        # Emit final problem
        if final_state.get("final_problem"):
            problem = final_state["final_problem"]
            yield _sse_event("problem_ready", {
                "problem_id": problem.problem_id,
                "latex_problem": problem.latex_problem,
                "image_base64": problem.image_base64,
                "drawing_path": final_state.get("drawing_path"),
                "params": problem.params.model_dump() if problem.params else {},
            })
        else:
            yield _sse_event("error", {"message": "题目生成失败"})

    # Use a simpler synchronous approach with streaming via thread
    import queue as _queue
    event_queue = _queue.Queue()

    async def simple_generator() -> AsyncGenerator[str, None]:
        graph = get_graph()
        loop = asyncio.get_event_loop()

        result_holder = {}

        def run_sync():
            try:
                state = dict(initial_state)
                final = graph.invoke(state)
                result_holder["final"] = final
            except Exception as e:
                import traceback
                result_holder["error"] = traceback.format_exc()

        task = loop.run_in_executor(None, run_sync)

        # Poll for completion while yielding heartbeats
        while not task.done():
            await asyncio.sleep(0.3)
            yield ": heartbeat\n\n"

        await task

        if "error" in result_holder:
            yield _sse_event("error", {"message": result_holder["error"]})
            return

        final = result_holder.get("final", {})

        # Stream reasoning trace
        for step in final.get("reasoning_trace", []):
            yield _sse_event("reasoning_step", {
                "step_id": step.step_id,
                "node_name": step.node_name,
                "action": step.action,
                "tool_called": step.tool_called,
                "tool_input_summary": step.tool_input_summary,
                "tool_output_summary": step.tool_output_summary,
                "drawing_path": step.drawing_path,
            })

        problem = final.get("final_problem")
        if problem:
            yield _sse_event("problem_ready", {
                "problem_id": problem.problem_id,
                "latex_problem": problem.latex_problem,
                "image_base64": problem.image_base64,
                "drawing_path": final.get("drawing_path"),
                "params": problem.params.model_dump() if problem.params else {},
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
async def solve_problem(req: SolveRequest):
    """Generate step-by-step solution for a problem (on demand)."""
    from models.problem import ProblemParams, ConicParams
    try:
        params = ProblemParams(**req.params) if req.params else None
    except Exception:
        params = None

    state = {
        "topic": req.params.get("topic", "ellipse") if req.params else "ellipse",
        "difficulty": req.params.get("difficulty", 3) if req.params else 3,
        "subtopics": [],
        "llm_config": req.llm_config,
        "latex_problem": req.latex_problem,
        "params": params,
        "reasoning_trace": [],
        "step_counter": 0,
    }

    result = solution_generation_node(state)
    return {"solution_latex": result.get("solution_latex", "")}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sse_event(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **data}, ensure_ascii=False)}\n\n"


def _run_graph_with_streaming(graph, initial_state, event_queue):
    """Run graph synchronously (called from thread executor)."""
    return graph.invoke(initial_state)
