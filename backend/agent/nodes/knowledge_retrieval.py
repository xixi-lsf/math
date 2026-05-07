"""
Node 1: Knowledge retrieval — ChromaDB + Tavily web search fallback.
根据用户输入的主题、难度、子主题，从一个向量数据库（ChromaDB）中检索相关的知识片段（公式、概念等），
并在检索结果不足时通过 Tavily 网络搜索 进行补充。
最后将这些知识片段存入状态，供后续题目生成节点使用
"""
from __future__ import annotations
import time
from agent.state import AgentState
from agent.logger import log_step, logger
from models.problem import ReasoningStep
from knowledge.vectordb import get_store


def knowledge_retrieval_node(state: AgentState) -> dict:
    #从状态中读取主题，难度，子主题列表，计数器
    topic = state["topic"]
    difficulty = state["difficulty"]
    subtopics = state.get("subtopics", [])
    step_id = state.get("step_counter", 0)
    selected_knowledge_ids: list[str] = state.get("selected_knowledge_ids", [])

    #查询字符串（eg.ellipse 解析几何 难度3 焦点 弦长）
    query = f"{topic} 解析几何 难度{difficulty} {' '.join(subtopics)}"
    store = get_store()

    # 1. ChromaDB retrieval — user knowledge (if selected) + builtin knowledge
    chunks = store.retrieve(
        query, topic, difficulty, n_results=6,
        selected_knowledge_ids=selected_knowledge_ids,
    )

    logger.info(
        "[knowledge_retrieval] topic=%s difficulty=%s total=%s user_knowledge_selected=%s",
        topic, difficulty, len(chunks), len(selected_knowledge_ids),
    )

    # 2. Tavily fallback if too few results
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

    #构建推理步骤
    action = f"从知识库检索到 {len(chunks)} 条相关知识"
    if web_used:
        action += "（含联网补充）"
    if selected_knowledge_ids:
        action += f"（含 {len(selected_knowledge_ids)} 份用户知识文档）"

    step = ReasoningStep(
        step_id=step_id,
        node_name="knowledge_retrieval",
        action=action,
        tool_called="ChromaDB.retrieve" + (" + Tavily.search" if web_used else ""),
        tool_input_summary=f"topic={topic}, difficulty={difficulty}",
        tool_output_summary=f"{len(chunks)} 条知识片段",
    )

    q = state.get("step_queue")
    if q is not None:
        q.put_nowait(step)
    log_step(step)

    return {
        "retrieved_knowledge": chunks,
        "reasoning_trace": state.get("reasoning_trace", []) + [step],
        "step_counter": step_id + 1,
    }
