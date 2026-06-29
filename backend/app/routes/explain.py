import os
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["explain"])

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
MODEL = "poolside/laguna-m.1:free"

RETINA_CLASS_INFO = {
    0: "No DR (No Diabetic Retinopathy) — normal healthy retina with no signs of damage",
    1: "Mild NPDR (Non-Proliferative Diabetic Retinopathy) — early stage with tiny bulges in blood vessels called microaneurysms",
    2: "Moderate NPDR — more widespread changes including dot hemorrhages and hard exudates (lipid deposits)",
    3: "Severe NPDR — significant blockage of retinal blood vessels causing reduced blood supply (retinal ischemia)",
    4: "Proliferative DR — advanced stage with abnormal new blood vessel growth on the retina surface, risk of vision loss",
}

SKIN_CLASS_INFO = {
    0: "Benign Nevus — a common non-cancerous mole, usually harmless",
    1: "Melanoma — a serious form of skin cancer that develops in melanocytes",
    2: "Basal Cell Carcinoma — the most common type of skin cancer, slow-growing and rarely spreads",
    3: "Actinic Keratosis — a rough, scaly patch on the skin caused by sun damage, can progress to cancer",
    4: "Squamous Cell Carcinoma — a common type of skin cancer that can spread if untreated",
    5: "Seborrheic Keratosis — a non-cancerous skin growth that looks waxy, common with age",
    6: "Dermatofibroma — a small, firm non-cancerous skin nodule",
}


class PredictionItem(BaseModel):
    class_id: int
    label: str
    confidence: float


class ExplainRequest(BaseModel):
    test_type: str = "retinopathy"
    predictions: list[PredictionItem]
    primary_diagnosis: PredictionItem


class ExplainResponse(BaseModel):
    explanation: str


@router.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="AI explanation service not configured")

    test_type = request.test_type
    primary = request.primary_diagnosis
    sorted_preds = sorted(request.predictions, key=lambda x: x.confidence, reverse=True)
    probs_str = "\n".join(f"- {p.label}: {p.confidence * 100:.1f}%" for p in sorted_preds)

    if test_type == "skin":
        class_info = SKIN_CLASS_INFO
        specialty = "dermatology (skin lesion classification)"
        body_part = "skin lesion"
    else:
        class_info = RETINA_CLASS_INFO
        specialty = "ophthalmology (diabetic retinopathy screening)"
        body_part = "retinal"

    class_ref = "\n".join(f"- {k}: {v}" for k, v in class_info.items())

    prompt = f"""You are a medical AI assistant specialized in {specialty}. A patient has uploaded a {body_part} image and the AI screening model produced these results:

Primary Diagnosis: {primary.label} ({primary.confidence * 100:.1f}% confidence)

All Probabilities:
{probs_str}

Class Reference:
{class_ref}

Please provide a clear, compassionate explanation for the patient in plain English using this exact structure. Use the section names as plain text (no bold, no markdown symbols):

Overview
What this result means in simple terms.

Symptoms
What symptoms a patient might or might not experience.

Causes
What causes this condition.

Risk Factors
Common risk factors that contribute to this condition.

Diagnosis
How this condition is diagnosed (including this AI screening).

Treatment Options
Available treatment approaches.

Prevention
Steps to prevent progression or maintain health.

Recovery Timeline
What to expect going forward.

Frequently Asked Questions
1-2 common questions patients have about this finding.

Related Diseases
Other conditions associated with this finding.

Keep the tone warm, educational, and reassuring. Do NOT use any markdown symbols like ** * or # anywhere in the output. Use plain text for section names and content. Limit each section to 2-3 sentences unless more detail is essential."""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            data = resp.json()
            if "error" in data:
                raise HTTPException(status_code=502, detail=data["error"].get("message", "AI model error"))
            explanation = data["choices"][0]["message"]["content"]
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="AI explanation timed out — please try again")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"AI service error: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return ExplainResponse(explanation=explanation)
