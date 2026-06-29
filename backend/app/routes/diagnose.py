import os
import time
import base64
import tempfile
from pathlib import Path
import httpx
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.app.schemas.diagnosis import DiagnosisResponse, DiagnosisResult
from backend.app.utils.image import preprocess_image, IMG_SIZE
from backend.app.models.retinopathy import RetinopathyModel, CLASS_LABELS
from backend.app.models.skin import SkinModel, SKIN_CLASS_LABELS

router = APIRouter(prefix="/api/v1", tags=["diagnosis"])

retina_model = RetinopathyModel()
skin_model = SkinModel()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
MAX_FILE_SIZE = 10 * 1024 * 1024
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()


async def validate_image(image_bytes: bytes, test_type: str) -> tuple[bool, str]:
    if not OPENROUTER_API_KEY:
        return True, ""
    if test_type == "skin":
        return True, ""
    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/jpeg;base64,{b64}"
    prompt = "Is this a valid retinal fundus photo of the human eye? Answer with exactly one word: 'yes' or 'no'. If no, add a brief reason in 5 words or less."

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "nvidia/nemotron-nano-12b-v2-vl:free",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }],
                },
            )
            data = resp.json()
            if "error" in data:
                return False, "image validation unavailable"
            answer = data["choices"][0]["message"]["content"].strip().lower()
            if answer.startswith("no"):
                reason = answer.replace("no", "", 1).strip().lstrip(",").strip() or "not a retinal image"
                return False, reason
            return True, ""
    except Exception:
        return False, "could not validate image"


@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(file: UploadFile = File(...), test_type: str = Form("retinopathy")):
    if test_type not in ("retinopathy", "skin"):
        raise HTTPException(status_code=400, detail=f"Unknown test type: {test_type}")

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

    is_valid, reason = await validate_image(contents, test_type)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"This doesn't appear to be a valid retinal image: {reason}. Please upload a clear photo of the retina.",
        )

    start = time.perf_counter()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        input_tensor = preprocess_image(tmp_path, IMG_SIZE)
        if test_type == "retinopathy":
            result = retina_model.predict(input_tensor)
        else:
            result = skin_model.predict(input_tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    elapsed = (time.perf_counter() - start) * 1000

    top_conf = result["primary_diagnosis"]["confidence"]
    if top_conf < 0.25:
        raise HTTPException(
            status_code=400,
            detail="The model is uncertain about this image (low confidence across all classes). Please upload a clear image.",
        )

    response = DiagnosisResponse(
        filename=file.filename,
        test_type=test_type,
        predictions=[DiagnosisResult(**p) for p in result["predictions"]],
        primary_diagnosis=DiagnosisResult(**result["primary_diagnosis"]),
        processing_time_ms=round(elapsed, 2),
    )

    return response


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "retina_loaded": retina_model.is_loaded,
        "skin_loaded": skin_model.is_loaded,
        "version": "1.0.0",
    }


@router.get("/labels")
async def get_labels(test_type: str = "retinopathy"):
    if test_type == "skin":
        return {"classes": SKIN_CLASS_LABELS}
    return {"classes": CLASS_LABELS}
