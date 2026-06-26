from fastapi import APIRouter, HTTPException
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
        raise HTTPException(statusCode=503, detail="Supabase not configured")
    try:
        resp = client.auth.sign_up({"email": req.email, "password": req.password})
        return {"user": resp.user.email if resp.user else None}
    except Exception as e:
        raise HTTPException(statusCode=400, detail=str(e))


@router.post("/signin")
async def sign_in(req: SignInRequest):
    client = get_client()
    if not client:
        raise HTTPException(statusCode=503, detail="Supabase not configured")
    try:
        resp = client.auth.sign_in_with_password(
            {"email": req.email, "password": req.password}
        )
        return {
            "user": resp.user.email if resp.user else None,
            "access_token": resp.session.access_token if resp.session else None,
        }
    except Exception as e:
        raise HTTPException(statusCode=401, detail=str(e))


@router.get("/me")
async def get_me(token: str = ""):
    client = get_client()
    if not client:
        raise HTTPException(statusCode=503, detail="Supabase not configured")
    try:
        if token:
            client.auth.set_session(token, "")
        resp = client.auth.get_user()
        return {"email": resp.user.email}
    except Exception as e:
        raise HTTPException(statusCode=401, detail=str(e))
