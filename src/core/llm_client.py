import os
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

def set_provider(provider: str, claude_key: str = "", model: Optional[str] = None) -> None:
    os.environ["LLM_PROVIDER"] = "claude"
    if claude_key:
        os.environ["ANTHROPIC_API_KEY"] = claude_key
    if model:
        os.environ["ANTHROPIC_MODEL"] = model


def current_settings() -> dict:
    return {
        "provider": "Claude",
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        "base_url": os.getenv("ANTHROPIC_BASE_URL", "").strip() or None,
    }


def chat_complete(messages: List[Dict[str, str]],
                  provider: Optional[str] = None,
                  api_key: Optional[str] = None,
                  model: Optional[str] = None,
                  temperature: float = 0.1,
                  base_url: Optional[str] = None,
                  **kwargs) -> str:
    cfg = current_settings()
    provider = provider or cfg["provider"]
    api_key = api_key or cfg["api_key"]
    model = model or cfg["model"]
    base_url = base_url or cfg.get("base_url")

    if (provider or "").lower() in {"claude", "anthropic"}:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key, base_url=base_url) if base_url else Anthropic(api_key=api_key)
            system_messages = [m["content"] for m in messages if m.get("role") == "system"]
            chat_messages = [m for m in messages if m.get("role") != "system"]
            resp = client.messages.create(
                model=model,
                max_tokens=kwargs.get("max_tokens", 1024),
                messages=chat_messages,
                temperature=temperature,
                system="\n\n".join(system_messages) if system_messages else None,
            )
            return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        except Exception as e:
            raise RuntimeError(f"Claude chat error: {e}")
    else:
        raise RuntimeError("No provider configured. Set LLM_PROVIDER=claude and ANTHROPIC_API_KEY.")