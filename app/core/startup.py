# app/core/startup.py

from app.ml.model_loader import model_bundle

def load_ai_model():
    """
    Load AI model at application startup
    """
    print("🚀 Loading Customer Support AI model...")
    model_bundle.load()
    print("✅ AI model loaded successfully")
