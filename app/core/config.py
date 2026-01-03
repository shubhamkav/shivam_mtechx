# app/core/config.py

from dotenv import load_dotenv
import os

load_dotenv()

APP_NAME = "Customer Support AI"
APP_VERSION = "1.0.0"

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/perfect_emotion_model.pth"
)
