# app/ml/response_generator.py

import random

RESPONSES = {
    "anger": [
        "I understand you're angry. Let me fix this immediately.",
        "I hear your frustration. I'm on it."
    ],
    "frustration": [
        "I know this is frustrating. Let me help.",
        "Thanks for your patience. Fixing this now."
    ],
    "neutral": [
        "Sure, I can help with that.",
        "Let me assist you."
    ],
    "urgency": [
        "This is urgent. Handling it now.",
        "Priority issue detected. Fixing immediately."
    ]
}

def generate_response(emotion: str):
    return random.choice(RESPONSES.get(emotion, RESPONSES["neutral"]))
