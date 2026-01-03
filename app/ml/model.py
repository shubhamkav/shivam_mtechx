# app/ml/model.py

import torch
import torch.nn as nn

class CustomerSupportAI(nn.Module):
    def __init__(self, vocab_size, num_categories, num_emotions):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 256, batch_first=True, bidirectional=True)

        self.category_head = nn.Linear(512, num_categories)
        self.emotion_head = nn.Linear(512, num_emotions)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        return self.category_head(pooled), self.emotion_head(pooled)
