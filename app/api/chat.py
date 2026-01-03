from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.ml.model_loader import model_bundle

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/", response_model=ChatResponse)
def chat(payload: ChatRequest):
    if not model_bundle.is_ready():
        raise HTTPException(status_code=503, detail="AI not loaded")

    response, category, emotion, confidence = model_bundle.respond(payload.message)

    return ChatResponse(
        response=response,
        category=category,
        emotion=emotion,
        confidence=confidence
    )
