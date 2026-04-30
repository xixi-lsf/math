"""
Node 1: Knowledge retrieval — ChromaDB + Tavily web search fallback.
"""
from __future__ import annotations
import time
from agent.state import AgentState
from models.problem import ReasoningStep
from knowledge.vectordb import get_store


def knowledge_retrieval_node(state: AgentState) -> dict:
    topic = state["topic"]
    difficulty = state["difficulty"]
    subtopics = state.get("subtopics", [])
    step_id = state.get("step_counter", 0)

    query = f"{topic} 解析几何 难度{difficulty} {' '.join(subtopics)}"
    store = get_store()

    # 1. ChromaDB retrieval
    chunks = store.retrieve(query, topic, difficulty, n_results=6)

    # 2. Also search user documents
    user_chunks = store.retrieve_user_docs(query, n_results=3)
    chunks = chunks + user_chunks

    # 3. Tavily fallback if too few results
    web_used = False
    if len(chunks) < 3:
        try:
            from tavily import TavilyClient
            import os
            api_key = os.getenv("TAVILY_API_KEY", "")
            if api_key:
                client = TavilyClient(api_key=api_key)
                results = client.search(
                    query=f"{topic} 解析几何定理公式 高考竞赛",
                    max_results=3,
                )
                from models.knowledge import KnowledgeChunk
                import hashlib
                for r in results.get("results", []):
                    chunk = KnowledgeChunk(
                        id=hashlib.md5(r["url"].encode()).hexdigest()[:12],
                        content=r.get("content", ""),
                        latex_formula="",
                        topic=topic,
                        subtopic="web_search",
                        source="web_search",
                    )
                    store.add_web_result(chunk)
                    chunks.append(chunk)
                web_used = True
        except Exception:
            pass

    action = f"从知识库检索到 {len(chunks)} 条相关知识"
    if web_used:
        action += "（含联网补充）"

    step = ReasoningStep(
        step_id=step_id,
        node_name="knowledge_retrieval",
        action=action,
        tool_called="ChromaDB.retrieve" + (" + Tavily.search" if web_used else ""),
        tool_input_summary=f"topic={topic}, difficulty={difficulty}",
        tool_output_summary=f"{len(chunks)} 条知识片段",
    )

    return {
        "retrieved_knowledge": chunks,
        "reasoning_trace": state.get("reasoning_trace", []) + [step],
        "step_counter": step_id + 1,
    }
