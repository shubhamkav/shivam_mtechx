# app/ml/model_loader.py

import torch
from app.ml.emotion_detector import EmotionDetector
from app.ml.response_generator import generate_response

EMOTIONS = ["anger", "frustration", "sadness", "urgency", "happiness", "neutral"]

class ModelBundle:
    def __init__(self):
        self.model = None
        self.vocab = {}
        self.emotion_detector = EmotionDetector()
        self.ready = False

    def load(self):
        checkpoint = torch.load(
            "models/perfect_emotion_model.pth",
            map_location="cpu"
        )

        self.vocab = checkpoint["vocab"]

        class EmotionModel(torch.nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.embed = torch.nn.Embedding(vocab_size + 1, 32, padding_idx=0)
                self.fc = torch.nn.Linear(32, 6)

            def forward(self, x):
                return self.fc(self.embed(x).mean(dim=1))

        self.model = EmotionModel(len(self.vocab))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.ready = True
        print("✅ ML model loaded successfully")

    def encode(self, text):
        tokens = text.lower().split()
        ids = [self.vocab.get(t, 0) for t in tokens]
        return torch.tensor([(ids + [0] * 6)[:6]])

    def respond(self, text: str):
        with torch.no_grad():
            logits = self.model(self.encode(text))
            emotion_idx = logits.argmax(dim=1).item()

        emotion = EMOTIONS[emotion_idx]
        response = generate_response(emotion)

        return response, "general", emotion, 0.85

    def is_ready(self):
        return self.ready

model_bundle = ModelBundle()
