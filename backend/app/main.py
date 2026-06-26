from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routes.diagnose import router as diagnose_router

app = FastAPI(
    title="RetinaScreen AI",
    description="AI-powered diabetic retinopathy detection from retinal fundus images",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(diagnose_router)


@app.get("/")
async def root():
    return {
        "service": "RetinaScreen AI",
        "version": "1.0.0",
        "endpoints": {
            "diagnose": "POST /api/v1/diagnose",
            "health": "GET /api/v1/health",
            "labels": "GET /api/v1/labels",
        },
    }
