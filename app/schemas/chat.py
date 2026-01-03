# app/schemas/chat.py

from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, example="I can't login and I'm frustrated")

class ChatResponse(BaseModel):
    response: str
    category: str
    emotion: str
    confidence: float
