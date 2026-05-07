"""
FastAPI application entry point.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.problems import router as problems_router
from api.knowledge import router as knowledge_router
from api.config import router as config_router
from knowledge.vectordb import get_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_store()
    store.load_example_problems()
    print("[startup] 知识库已加载")
    yield


app = FastAPI(
    title="解析几何题目生成系统",
    description="基于 LangGraph Agent 的智能解析几何题目生成器",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(problems_router)
app.include_router(knowledge_router)
app.include_router(config_router)


@app.get("/")
async def root():
    return {"message": "解析几何题目生成系统 API", "docs": "/docs"}
