import os
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["explain"])

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
MODEL = "poolside/laguna-m.1:free"

CLASS_INFO = {
    0: "No DR (No Diabetic Retinopathy) — normal healthy retina with no signs of damage",
    1: "Mild NPDR (Non-Proliferative Diabetic Retinopathy) — early stage with tiny bulges in blood vessels called microaneurysms",
    2: "Moderate NPDR — more widespread changes including dot hemorrhages and hard exudates (lipid deposits)",
    3: "Severe NPDR — significant blockage of retinal blood vessels causing reduced blood supply (retinal ischemia)",
    4: "Proliferative DR — advanced stage with abnormal new blood vessel growth on the retina surface, risk of vision loss",
}


class PredictionItem(BaseModel):
    class_id: int
    label: str
    confidence: float


class ExplainRequest(BaseModel):
    predictions: list[PredictionItem]
    primary_diagnosis: PredictionItem


class ExplainResponse(BaseModel):
    explanation: str


@router.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="AI explanation service not configured")

    primary = request.primary_diagnosis
    sorted_preds = sorted(request.predictions, key=lambda x: x.confidence, reverse=True)
    probs_str = "\n".join(f"- {p.label}: {p.confidence * 100:.1f}%" for p in sorted_preds)
    class_ref = "\n".join(f"- {k}: {v}" for k, v in CLASS_INFO.items())

    prompt = f"""You are a medical AI assistant specialized in diabetic retinopathy screening. A patient has uploaded a retinal image and the AI screening model produced these results:

**Primary Diagnosis**: {primary.label} ({primary.confidence * 100:.1f}% confidence)

**All Probabilities**:
{probs_str}

**Class Reference**:
{class_ref}

Please provide a clear, compassionate explanation for the patient in plain English using this exact structure with bold section headings:

**Overview**
What this result means in simple terms.

**Symptoms**
What symptoms a patient might or might not experience at this stage.

**Causes**
What causes this stage of diabetic retinopathy.

**Risk Factors**
Common risk factors that contribute to this condition.

**Diagnosis**
How this condition is diagnosed (including this AI screening).

**Treatment Options**
Available treatment approaches for this stage.

**Prevention**
Steps to prevent progression or maintain eye health.

**Recovery Timeline**
What to expect going forward.

**Frequently Asked Questions**
1-2 common questions patients have about this finding.

**Related Diseases**
Other conditions associated with diabetic retinopathy.

Keep the tone warm, educational, and reassuring. Use bold for each heading (e.g., **Overview**). Limit each section to 2-3 sentences unless more detail is essential."""

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
