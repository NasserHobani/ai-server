"""Uvicorn entry for the FastAPI LLM proxy (separate from FastMCP)."""

from __future__ import annotations

import os


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    import uvicorn

    uvicorn.run(
        "cs_ai_bridge_api.app:app",
        host="0.0.0.0",
        port=int(os.getenv("CS_AI_BRIDGE_API_PORT", "8080")),
        factory=False,
    )


if __name__ == "__main__":
    main()
