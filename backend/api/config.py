"""
FastAPI routes for LLM configuration validation.
"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/config", tags=["config"])


class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model: str


@router.post("/validate-llm")
async def validate_llm(config: LLMConfig):
    """Test if the provided LLM config is reachable and working."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": "回复 OK"}],
            max_tokens=10,
            temperature=0,
        )
        reply = response.choices[0].message.content.strip()
        return {"valid": True, "reply": reply, "model": config.model}
    except Exception as e:
        return {"valid": False, "error": str(e)}
