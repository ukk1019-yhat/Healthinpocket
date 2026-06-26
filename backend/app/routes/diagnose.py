import time
import tempfile
import base64
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from backend.app.schemas.diagnosis import DiagnosisResponse, DiagnosisResult
from backend.app.utils.image import preprocess_image, IMG_SIZE
from backend.app.models.retinopathy import RetinopathyModel, CLASS_LABELS
from backend.app.db import get_client, is_configured

router = APIRouter(prefix="/api/v1", tags=["diagnosis"])
model = RetinopathyModel()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
MAX_FILE_SIZE = 10 * 1024 * 1024


@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(file: UploadFile = File(...), authorization: str = Header(None)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, detail="File too large (max 10MB)"
        )
    start = time.perf_counter()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        input_tensor = preprocess_image(tmp_path, IMG_SIZE)
        result = model.predict(input_tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    elapsed = (time.perf_counter() - start) * 1000

    response = DiagnosisResponse(
        filename=file.filename,
        predictions=[DiagnosisResult(**p) for p in result["predictions"]],
        primary_diagnosis=DiagnosisResult(**result["primary_diagnosis"]),
        processing_time_ms=round(elapsed, 2),
    )

    # Store in Supabase if configured and authenticated
    _store_screening(contents, result, elapsed, authorization)

    return response


def _store_screening(image_bytes: bytes, result: dict, elapsed: float, auth_header: str | None):
    if not is_configured():
        return
    client = get_client()
    if not auth_header:
        return
    token = auth_header.replace("Bearer ", "")
    try:
        client.auth.set_session(token, "")
        client.table("screenings").insert({
            "test_type": "retinopathy",
            "filename": "upload",
            "image_b64": base64.b64encode(image_bytes).decode(),
            "primary_diagnosis": result["primary_diagnosis"]["label"],
            "primary_confidence": result["primary_diagnosis"]["confidence"],
            "all_predictions": result["predictions"],
            "processing_time_ms": round(elapsed, 2),
        }).execute()
    except Exception:
        pass


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model.is_loaded,
        "version": "1.0.0",
    }


@router.get("/labels")
async def get_labels():
    return {"classes": CLASS_LABELS}
