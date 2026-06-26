from fastapi import APIRouter, HTTPException, Header
from backend.app.db import get_client

router = APIRouter(prefix="/api/v1/history", tags=["history"])


@router.get("/")
async def get_history(authorization: str = Header(None)):
    client = get_client()
    if not client:
        raise HTTPException(statusCode=503, detail="Supabase not configured")
    if not authorization:
        raise HTTPException(statusCode=401, detail="Missing authorization header")
    token = authorization.replace("Bearer ", "")
    try:
        client.auth.set_session(token, "")
        resp = (
            client.table("screenings")
            .select("*")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        return {"screenings": resp.data}
    except Exception as e:
        raise HTTPException(statusCode=401, detail=str(e))
