import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from backend.app.db import get_client

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class SignUpRequest(BaseModel):
    email: str
    password: str


class SignInRequest(BaseModel):
    email: str
    password: str


@router.post("/signup")
async def sign_up(req: SignUpRequest):
    client = get_client()
    if not client:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        resp = client.auth.sign_up({"email": req.email, "password": req.password})
        return {"user": resp.user.email if resp.user else None}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/signin")
async def sign_in(req: SignInRequest):
    client = get_client()
    if not client:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        resp = client.auth.sign_in_with_password(
            {"email": req.email, "password": req.password}
        )
        return {
            "user": resp.user.email if resp.user else None,
            "access_token": resp.session.access_token if resp.session else None,
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.get("/oauth/{provider}")
async def oauth_login(provider: str, request: Request):
    client = get_client()
    if not client:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        callback = str(request.url_for("oauth_callback"))
        supabase_url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
        # Manually build implicit grant URL (access_token in hash, not PKCE code)
        auth_url = (
            f"{supabase_url}/auth/v1/authorize"
            f"?provider={provider}"
            f"&redirect_to={callback}"
        )
        if request.base_url.scheme == "https":
            auth_url += "&response_type=token"
        return {"url": auth_url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


OAUTH_CALLBACK_HTML = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Signing in...</title></head>
<body>
<script>
try {
  var h = new URLSearchParams(window.location.hash.substring(1));
  var token = h.get("access_token");
  var email = h.get("email") || "";
  if (token && window.opener) {
    window.opener.postMessage({ type: "oauth", token: token, email: email }, "*");
  }
} catch(e) {}
window.close();
document.body.innerHTML = "<p>Signed in! You may close this window.</p>";
</script>
</body>
</html>"""

@router.get("/callback", response_class=HTMLResponse, name="oauth_callback")
async def oauth_callback():
    return OAUTH_CALLBACK_HTML


@router.post("/exchange")
async def exchange_code(code: str = ""):
    client = get_client()
    if not client or not code:
        raise HTTPException(status_code=400, detail="Missing code")
    try:
        resp = client.auth.exchange_code_for_session({"auth_code": code})
        return {
            "access_token": resp.session.access_token if resp.session else None,
            "user": resp.user.email if resp.user else None,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/me")
async def get_me(token: str = ""):
    client = get_client()
    if not client:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        if token:
            client.auth.set_session(token, token)
        resp = client.auth.get_user()
        return {"email": resp.user.email}
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))
