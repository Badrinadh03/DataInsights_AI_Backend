import os
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

def set_provider(provider: str, mistral_key: str = "", openai_key: str = "", model: Optional[str] = None) -> None:
    p = (provider or "None").strip()
    os.environ["LLM_PROVIDER"] = p
    if p.lower() == "mistral" and mistral_key:
        os.environ["MISTRAL_API_KEY"] = mistral_key
        if model:
            os.environ["MISTRAL_MODEL"] = model
    elif p.lower() == "openai" and openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        if model:
            os.environ["OPENAI_MODEL"] = model

def current_settings() -> dict:
    provider = (os.getenv("LLM_PROVIDER", "OpenAI")).strip()
    if provider.lower() == "mistral":
        return {
            "provider": "Mistral",
            "api_key": os.getenv("MISTRAL_API_KEY", ""),
            "model": os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
            "base_url": os.getenv("MISTRAL_BASE_URL", "").strip() or None,
        }
    elif provider.lower() == "openai":
        return {
            "provider": "OpenAI",
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "base_url": os.getenv("OPENAI_BASE_URL", "").strip() or None,
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
    api_key  = api_key or cfg["api_key"]
    model    = model or cfg["model"]
    base_url = base_url or cfg.get("base_url")

    if (provider or "").lower() == "openai":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI chat error: {e}")
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
    else:
        raise RuntimeError("No provider configured. Set LLM_PROVIDER or call set_provider().")