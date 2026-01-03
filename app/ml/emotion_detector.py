# app/ml/emotion_detector.py

from collections import defaultdict

class EmotionDetector:
    def __init__(self):
        self.emotions = {
            "anger": ["angry", "mad", "furious", "hate", "annoyed"],
            "frustration": ["frustrated", "fed up", "irritated"],
            "sadness": ["sad", "disappointed", "upset"],
            "urgency": ["urgent", "asap", "immediately", "emergency"],
            "happiness": ["happy", "glad", "satisfied"]
        }

    def detect(self, text: str):
        text = text.lower()
        scores = defaultdict(int)

        for emotion, words in self.emotions.items():
            for w in words:
                if w in text:
                    scores[emotion] += 1

        if not scores:
            return "neutral", 0.5

        emotion = max(scores, key=scores.get)
        confidence = min(0.9, 0.5 + scores[emotion] * 0.1)
        return emotion, confidence
