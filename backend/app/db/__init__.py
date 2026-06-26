import os
from supabase import create_client, Client

_url = os.environ.get("SUPABASE_URL", "")
_key = os.environ.get("SUPABASE_ANON_KEY", "")

_client: Client | None = None


def get_client() -> Client | None:
    global _client
    if _client is None and _url and _key:
        _client = create_client(_url, _key)
    return _client


def is_configured() -> bool:
    return bool(_url and _key)
