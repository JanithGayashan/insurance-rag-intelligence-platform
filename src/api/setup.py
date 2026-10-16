"""Create the private backend token without displaying it or changing other keys."""
from __future__ import annotations

import secrets

from dotenv import dotenv_values, set_key

from src.generation.configuration import PROJECT_ROOT


def main():
    path = PROJECT_ROOT / ".env"
    existing = dotenv_values(path).get("RAG_BACKEND_API_KEY")
    if existing:
        print("Existing RAG_BACKEND_API_KEY preserved in local .env.")
        return
    path.touch(exist_ok=True)
    set_key(path, "RAG_BACKEND_API_KEY", secrets.token_urlsafe(32))
    print("Created RAG_BACKEND_API_KEY in local .env. Its value was not displayed.")


if __name__ == "__main__":
    main()
