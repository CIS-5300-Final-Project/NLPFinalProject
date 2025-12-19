import torch
import torch.nn as nn

class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        out = self.token_emb(x) + self.pos_emb(positions)
        return self.norm(out)

class TransformerLayer(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4*hidden),
            nn.ReLU(),
            nn.Linear(4*hidden, hidden)
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class MiniBERT(nn.Module):
    def __init__(self, vocab_size, hidden=256, layers=4, heads=4, num_labels=7):
        super().__init__()
        self.embed = BertEmbedding(vocab_size, hidden)
        self.layers = nn.ModuleList([TransformerLayer(hidden, heads) for _ in range(layers)])
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        cls_token = x[:, 0, :]
        return self.classifier(cls_token)
