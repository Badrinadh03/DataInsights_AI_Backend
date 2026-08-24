import os
import sys
import importlib.util
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass


def _real_anthropic_module():
    """
    Vercel's Python runtime pre-imports its own internal package also named
    "anthropic" (under /var/task/_vendor) before our code ever runs, which caches
    it in sys.modules under the "anthropic" key. Reordering sys.path doesn't help
    once a module name is cached, so `import anthropic` anywhere in this process
    keeps returning that shim (whose Messages.create() has a different signature)
    instead of the real SDK installed from requirements.txt.

    This loads the real package straight from site-packages by file path and
    overwrites sys.modules["anthropic"] with it, so every subsequent import in
    this process (including plain `import anthropic`) resolves to the real SDK.
    No-op if the real module is already the one loaded, or outside this runtime.
    """
    module = sys.modules.get("anthropic")
    if module is not None and "_vendor" not in (getattr(module, "__file__", "") or ""):
        return module

    for entry in sys.path:
        if not entry.rstrip(os.sep).endswith("site-packages"):
            continue
        pkg_dir = os.path.join(entry, "anthropic")
        init_file = os.path.join(pkg_dir, "__init__.py")
        if os.path.isfile(init_file):
            spec = importlib.util.spec_from_file_location(
                "anthropic", init_file, submodule_search_locations=[pkg_dir]
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["anthropic"] = module
            spec.loader.exec_module(module)
            return module

    import anthropic as fallback
    return fallback

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
            Anthropic = _real_anthropic_module().Anthropic
            client = Anthropic(api_key=api_key, base_url=base_url) if base_url else Anthropic(api_key=api_key)
            system_messages = [m["content"] for m in messages if m.get("role") == "system"]
            chat_messages = [m for m in messages if m.get("role") != "system"]
            create_kwargs = {
                "model": model,
                "max_tokens": kwargs.get("max_tokens", 1024),
                "messages": chat_messages,
                "temperature": temperature,
            }
            if system_messages:
                create_kwargs["system"] = "\n\n".join(system_messages)
            resp = client.messages.create(**create_kwargs)
            return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        except Exception as e:
            raise RuntimeError(f"Claude chat error: {e}")
    else:
        raise RuntimeError("No provider configured. Set LLM_PROVIDER=claude and ANTHROPIC_API_KEY.")