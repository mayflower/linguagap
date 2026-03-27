"""Demo authentication for LinguaGap desktop interface."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from starlette.requests import Request

logger = logging.getLogger(__name__)


@dataclass
class DemoAccount:
    username: str
    password: str
    display_name: str
    logo_url: str


_accounts: list[DemoAccount] | None = None


def _load_accounts() -> list[DemoAccount]:
    """Load demo accounts from DEMO_ACCOUNTS env var or demo_accounts.json."""
    env_accounts = os.getenv("DEMO_ACCOUNTS")
    if env_accounts:
        raw = json.loads(env_accounts)
    else:
        config_path = Path(__file__).parent / "demo_accounts.json"
        if config_path.exists():
            raw = json.loads(config_path.read_text())
        else:
            logger.warning("No demo accounts configured")
            return []
    return [DemoAccount(**a) for a in raw]


def get_accounts() -> list[DemoAccount]:
    global _accounts  # noqa: PLW0603
    if _accounts is None:
        _accounts = _load_accounts()
    return _accounts


def verify_credentials(username: str, password: str) -> DemoAccount | None:
    for account in get_accounts():
        if account.username == username and account.password == password:
            return account
    return None


def get_current_user(request: Request) -> dict | None:
    username = request.session.get("username")
    if username:
        return {
            "username": username,
            "display_name": request.session.get("display_name", ""),
            "logo_url": request.session.get("logo_url", ""),
        }
    return None
