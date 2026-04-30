from pydantic import BaseModel
from typing import Optional


class KnowledgeChunk(BaseModel):
    id: str
    content: str           # natural language description
    latex_formula: str     # corresponding LaTeX
    topic: str             # "ellipse" | "hyperbola" | "parabola" | "polar"
    subtopic: str          # "focal_chord" | "eccentricity" | "asymptote" | ...
    difficulty_range: tuple[int, int] = (1, 5)
    source: str = "builtin"  # "builtin" | "user_upload" | "web_search"
    metadata: dict = {}
