"""
Node 1: Knowledge retrieval — ChromaDB + Tavily web search fallback.
根据用户输入的主题、难度、子主题，从一个向量数据库（ChromaDB）中检索相关的知识片段（公式、概念等），
并在检索结果不足时通过 Tavily 网络搜索 进行补充。
最后将这些知识片段存入状态，供后续题目生成节点使用
"""
from __future__ import annotations
import time
from agent.state import AgentState
from models.problem import ReasoningStep
from knowledge.vectordb import get_store


def knowledge_retrieval_node(state: AgentState) -> dict:
    #从状态中读取主题，难度，子主题列表，计数器
    topic = state["topic"]
    difficulty = state["difficulty"]
    subtopics = state.get("subtopics", [])
    step_id = state.get("step_counter", 0)

    #查询字符串（eg.ellipse 解析几何 难度3 焦点 弦长）
    #查询字符串会被向量化并与知识库中的文档进行相似度匹配。
    query = f"{topic} 解析几何 难度{difficulty} {' '.join(subtopics)}"
    #store = get_store()
    store = get_store()

    # 1. ChromaDB retrieval,调用 store.retrieve，执行向量相似度搜索,返回最多 6 个最相关的知识片段（
    chunks = store.retrieve(query, topic, difficulty, n_results=6)

    # 2. Also search user documents 检索用户自定义文档
    user_chunks = store.retrieve_user_docs(query, n_results=3)
    chunks = chunks + user_chunks

    # 3. Tavily fallback if too few results若结果不足，触发 Tavily 网络搜索（备用）
    web_used = False
    #前面两步总共得到的知识片段少于 3 条，则认为本地知识库覆盖面不足
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
