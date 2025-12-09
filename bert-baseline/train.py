import torch
from torch.utils.data import DataLoader
from bert_model import MiniBERT
from meld_dataset import MELDDataset
import sentencepiece as spm
from sklearn.metrics import accuracy_score
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("bert_sp.model")
vocab_size = sp.get_piece_size()

# Load datasets
train_ds = MELDDataset("data/MELD.Raw/train_sent_emo.csv", "bert_sp.model")
dev_ds = MELDDataset("data/MELD.Raw/dev_sent_emo.csv", "bert_sp.model")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=64)

# Model + optimizer
model = MiniBERT(vocab_size=vocab_size, num_labels=7).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = torch.nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item()

            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(y.cpu().tolist())

    acc = accuracy_score(labels, preds)
    avg_loss = total_loss / len(loader)
    return avg_loss, acc


def per_label_accuracy(model, loader, num_labels=7):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    per_label_acc = {}
    for label in range(num_labels):
        idx = (all_labels == label)
        if idx.sum() == 0:
            acc = None
        else:
            acc = accuracy_score(all_labels[idx], all_preds[idx])
        per_label_acc[label] = {
            "accuracy": acc,
            "support": idx.sum()
        }

    overall_acc = accuracy_score(all_labels, all_preds)

    return per_label_acc, overall_acc


# Training loop
for epoch in range(3):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    val_loss, val_acc = evaluate(model, dev_loader)

    print(f"Epoch {epoch}: "
          f"train_loss={train_loss:.4f} | "
          f"val_loss={val_loss:.4f} | "
          f"val_acc={val_acc:.4f}")

    per_label, overall = per_label_accuracy(model, dev_loader)

    print("\n=== Per-label Accuracy ===")
    emotion_names = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]

    for i, emo in enumerate(emotion_names):
        acc = per_label[i]["accuracy"]
        support = per_label[i]["support"]
        print(f"{emo:10s} | acc={acc:.4f} | count={support}")

    print(f"\nOverall accuracy: {overall:.4f}")

torch.save(model.state_dict(), "scratch_meld_bert.pt")
