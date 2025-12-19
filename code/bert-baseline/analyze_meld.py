import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
meld_files = {
    "train": "data/MELD.Raw/train_sent_emo.csv",
    "dev": "data/MELD.Raw/dev_sent_emo.csv",
    "test": "data/MELD.Raw/test_sent_emo.csv"
}

emotion2id = {
    "neutral": 0, "joy": 1, "surprise": 2,
    "anger": 3, "sadness": 4, "disgust": 5, "fear": 6
}

# --- LOAD DATA ---
def load_meld():
    datasets = {}
    for split, file in meld_files.items():
        df = pd.read_csv(file)
        datasets[split] = df
        print(f"Loaded {file} with {len(df)} samples")
    return datasets

datasets = load_meld()

# Combine for global stats
df_all = pd.concat(datasets.values(), ignore_index=True)

# --- BASIC INFO ---
print("\n=== Basic Dataset Info ===")
print(df_all.info())

print("\n=== Sample Rows ===")
print(df_all.head())

# --- EMOTION DISTRIBUTION ---
print("\n=== Emotion Distribution ===")
emotion_counts = df_all["Emotion"].value_counts().sort_index()
print(emotion_counts)

plt.figure(figsize=(10,5))
sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette="viridis")
plt.title("MELD Emotion Distribution")
plt.ylabel("Count")
plt.xlabel("Emotion")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("emotion_distribution.png")
plt.close()

# --- UTTERANCE LENGTH ANALYSIS ---
df_all["length"] = df_all["Utterance"].astype(str).apply(lambda x: len(x.split()))

print("\n=== Length Statistics ===")
print(df_all["length"].describe())

plt.figure(figsize=(10,5))
sns.histplot(df_all["length"], bins=40, kde=True, color="steelblue")
plt.title("Utterance Length Distribution")
plt.xlabel("Length (words)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("utterance_length_hist.png")
plt.close()

# --- LENGTH PER EMOTION ---
emotion_length = df_all.groupby("Emotion")["length"].mean().sort_values()
print("\n=== Average Utterance Length per Emotion ===")
print(emotion_length)

plt.figure(figsize=(10,5))
sns.barplot(x=emotion_length.index, y=emotion_length.values, palette="magma")
plt.title("Average Utterance Length per Emotion")
plt.ylabel("Avg Words")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("avg_length_per_emotion.png")
plt.close()

# --- SPEAKER DISTRIBUTION (optional) ---
if "Speaker" in df_all.columns:
    speaker_counts = df_all["Speaker"].value_counts().head(15)
    print("\n=== Top Speakers ===")
    print(speaker_counts)

    plt.figure(figsize=(10,6))
    sns.barplot(x=speaker_counts.values, y=speaker_counts.index, palette="coolwarm")
    plt.title("Top 15 Speakers in MELD")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig("speaker_distribution.png")
    plt.close()

# --- SPLIT STATISTICS ---
print("\n=== Split Statistics ===")
for split, df in datasets.items():
    print(f"{split.upper()}: {len(df)} samples")
    print(df["Emotion"].value_counts())
    print()

print("\nAnalysis complete! Plots saved:")
print("  - emotion_distribution.png")
print("  - utterance_length_hist.png")
print("  - avg_length_per_emotion.png")
print("  - speaker_distribution.png (if applicable)")
