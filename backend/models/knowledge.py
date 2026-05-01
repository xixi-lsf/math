from pydantic import BaseModel
from typing import Optional

"""
数据模型文件
知识片段模型：用于表示从知识库中检索到的一条知识条目
web检索的也会整合为这种形式
"""
class KnowledgeChunk(BaseModel):
    id: str
    content: str           # 自然语言描述的知识内容
    latex_formula: str     # 对应的 LaTeX 公式
    topic: str             # "ellipse" | "hyperbola" | "parabola" | "polar"
    subtopic: str          # "focal_chord" | "eccentricity" | "asymptote" | ...
    difficulty_range: tuple[int, int] = (1, 5)#该知识适用的难度范围
    source: str = "builtin"  #来源 "builtin" | "user_upload" | "web_search"
    metadata: dict = {}
