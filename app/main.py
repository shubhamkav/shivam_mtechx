# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import APP_NAME, APP_VERSION
from app.core.startup import load_ai_model
from app.api.chat import router as chat_router


# =========================
# LIFESPAN (Startup / Shutdown)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_ai_model()
    yield
    # Shutdown (optional cleanup later)


# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan
)


# =========================
# CORS CONFIG
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for college & dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# ROUTERS
# =========================
app.include_router(chat_router)


# =========================
# HEALTH CHECK
# =========================
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "running",
        "message": "Customer Support AI backend is live 🚀"
    }
