import os

_url = os.environ.get("SUPABASE_URL", "").strip()
_key = os.environ.get("SUPABASE_ANON_KEY", "").strip()

_client = None


def get_client():
    global _client
    if _client is None and _url and _key:
        try:
            from supabase import create_client
            _client = create_client(_url, _key)
        except ImportError:
            pass
    return _client


def is_configured() -> bool:
    return bool(_url and _key)
