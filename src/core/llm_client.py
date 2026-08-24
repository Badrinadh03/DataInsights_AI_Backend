import os
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

def set_provider(provider: str, mistral_key: str = "", claude_key: str = "", gemini_key: str = "", model: Optional[str] = None) -> None:
    p = (provider or "None").strip()
    os.environ["LLM_PROVIDER"] = p
    if p.lower() == "mistral" and mistral_key:
        os.environ["MISTRAL_API_KEY"] = mistral_key
        if model:
            os.environ["MISTRAL_MODEL"] = model
    elif p.lower() in {"claude", "anthropic"} and claude_key:
        os.environ["ANTHROPIC_API_KEY"] = claude_key
        if model:
            os.environ["ANTHROPIC_MODEL"] = model
    elif p.lower() in {"gemini", "google"} and gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        if model:
            os.environ["GEMINI_MODEL"] = model


def current_settings() -> dict:
    provider = (os.getenv("LLM_PROVIDER", "gemini")).strip()
    if provider.lower() == "mistral":
        return {
            "provider": "Mistral",
            "api_key": os.getenv("MISTRAL_API_KEY", ""),
            "model": os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
            "base_url": os.getenv("MISTRAL_BASE_URL", "").strip() or None,
        }
    elif provider.lower() in {"claude", "anthropic"}:
        return {
            "provider": "Claude",
            "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
            "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
            "base_url": os.getenv("ANTHROPIC_BASE_URL", "").strip() or None,
        }
    elif provider.lower() in {"gemini", "google"}:
        return {
            "provider": "Gemini",
            "api_key": os.getenv("GEMINI_API_KEY", ""),
            "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            "base_url": None,
        }
    return {"provider": "None", "api_key": "", "model": "", "base_url": None}


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
    elif (provider or "").lower() == "mistral":
        try:
            from mistralai.client import MistralClient
            from mistralai.models.chat_completion import ChatMessage
            client = MistralClient(api_key=api_key)
            msgs = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]
            resp = client.chat(model=model, messages=msgs, temperature=temperature)
            return resp.choices[0].message.content or ""
        except Exception:
            try:
                from mistralai import Mistral
                client = Mistral(api_key=api_key)
                resp = client.chat.complete(model=model, messages=messages, temperature=temperature)
                return resp.choices[0].message["content"]
            except Exception as e2:
                raise RuntimeError(f"Mistral chat error: {e2}")
    elif (provider or "").lower() in {"gemini", "google"}:
        try:
            from google import genai
            if not api_key:
                raise ValueError("GEMINI_API_KEY is missing")
            client = genai.Client(api_key=api_key)
            prompt = "\n\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
            resp = client.models.generate_content(
                model=model or "gemini-2.0-flash",
                contents=prompt,
            )
            return getattr(resp, "text", "") or ""
        except Exception as e:
            raise RuntimeError(f"Gemini chat error: {e}")
    else:
        raise RuntimeError("No provider configured. Set LLM_PROVIDER or call set_provider().")