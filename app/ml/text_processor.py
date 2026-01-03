# app/ml/text_processor.py

import re
import pickle

class TextProcessor:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.max_len = 25

    def preprocess(self, text):
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def encode(self, text):
        text = self.preprocess(text)
        tokens = text.split()
        indices = []

        for token in tokens:
            indices.append(self.word2idx.get(token, 1))

        if len(indices) > self.max_len:
            return indices[:self.max_len]

        return indices + [0] * (self.max_len - len(indices))

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.word2idx = data["word2idx"]
            self.idx2word = data["idx2word"]
