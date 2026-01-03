# train_model.py

import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# SIMPLE TRAINING DATA
# -----------------------------
texts = [
    "hello",
    "hi",
    "i am angry",
    "this is frustrating",
    "urgent help needed",
    "i am very happy",
]

labels_emotion = [
    5,  # neutral
    5,  # neutral
    0,  # anger
    1,  # frustration
    3,  # urgency
    4,  # happiness
]

# emotion index mapping
# 0 anger
# 1 frustration
# 2 sadness
# 3 urgency
# 4 happiness
# 5 neutral

# -----------------------------
# SIMPLE TEXT ENCODER
# -----------------------------
vocab = {word: i + 1 for i, word in enumerate(set(" ".join(texts).split()))}
PAD = 0
MAX_LEN = 6

def encode(text):
    tokens = text.split()
    ids = [vocab.get(t, 0) for t in tokens]
    return (ids + [PAD] * MAX_LEN)[:MAX_LEN]

X = torch.tensor([encode(t) for t in texts])
y = torch.tensor(labels_emotion)

# -----------------------------
# MODEL
# -----------------------------
class EmotionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, 32, padding_idx=0)
        self.fc = nn.Linear(32, 6)

    def forward(self, x):
        x = self.embed(x).mean(dim=1)
        return self.fc(x)

model = EmotionModel(len(vocab))

# -----------------------------
# TRAINING
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("🚀 Training model...")

for epoch in range(200):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

print("✅ Training complete")

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab": vocab
    },
    "models/perfect_emotion_model.pth"
)

print("💾 Model saved to models/perfect_emotion_model.pth")
