import torch
import pandas as pd
from torch.utils.data import Dataset
import sentencepiece as spm

class MELDDataset(Dataset):
    def __init__(self, csv_path, vocab_model, max_len=64):
        self.df = pd.read_csv(csv_path)
        self.utterances = self.df["Utterance"].tolist()
        self.labels = self.df["Emotion"].apply(self.emotion_to_id).tolist()

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(vocab_model)
        self.max_len = max_len

    def emotion_to_id(self, label):
        mapping = {
            "neutral": 0, "joy": 1, "surprise": 2,
            "anger": 3, "sadness": 4, "disgust": 5, "fear": 6
        }
        return mapping[label]

    def encode(self, text):
        ids = self.sp.encode(text, out_type=int)
        ids = ids[:self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        return self.encode(self.utterances[idx]), torch.tensor(self.labels[idx])
