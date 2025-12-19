import pandas as pd

# ---- LOAD MELD ----
meld_train = pd.read_csv("data/MELD.Raw/train_sent_emo.csv")
meld_dev = pd.read_csv("data/MELD.Raw/dev_sent_emo.csv")
meld_test = pd.read_csv("data/MELD.Raw/test_sent_emo.csv")

# Extract utterances
meld_text = (
    meld_train["Utterance"].tolist() +
    meld_dev["Utterance"].tolist() +
    meld_test["Utterance"].tolist()
)

# ---- LOAD PRESIDENTIAL SPEECH DATA ----
pres = pd.read_excel("data/1presidential_speeches_with_metadata.xlsx")
speech_col = "speech"

pres_text = pres[speech_col].dropna().tolist()

# ---- COMBINE ----
all_text = meld_text + pres_text

# ---- WRITE OUT ----
with open("all_text.txt", "w", encoding="utf-8") as f:
    for line in all_text:
        f.write(str(line).strip() + "\n")

print("Saved all_text.txt with", len(all_text), "lines")
