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
        redirect_to = str(request.url_for("oauth_callback"))
        resp = client.auth.sign_in_with_oauth(
            {"provider": provider, "options": {"redirect_to": redirect_to}}
        )
        return {"url": resp.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


OAUTH_CALLBACK_HTML = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Signing in...</title></head>
<body>
<script>
try {
  var hash = window.location.hash.substring(1);
  var params = new URLSearchParams(hash);
  var token = params.get("access_token");
  var email = params.get("email") || "";
  if (token) {
    window.opener.postMessage({ type: "oauth", access_token: token, email: email }, "*");
  }
} catch(e) {}
window.close();
</script>
</body>
</html>"""


@router.get("/callback", response_class=HTMLResponse, name="oauth_callback")
async def oauth_callback():
    return OAUTH_CALLBACK_HTML


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
