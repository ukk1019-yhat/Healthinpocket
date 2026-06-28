import time
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.schemas.diagnosis import DiagnosisResponse, DiagnosisResult
from backend.app.utils.image import preprocess_image, IMG_SIZE
from backend.app.models.retinopathy import RetinopathyModel, CLASS_LABELS

router = APIRouter(prefix="/api/v1", tags=["diagnosis"])
model = RetinopathyModel()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
MAX_FILE_SIZE = 10 * 1024 * 1024


@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(file: UploadFile = File(...)):
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

    return response
async def health():
    return {
        "status": "ok",
        "model_loaded": model.is_loaded,
        "version": "1.0.0",
    }


@router.get("/labels")
async def get_labels():
    return {"classes": CLASS_LABELS}
