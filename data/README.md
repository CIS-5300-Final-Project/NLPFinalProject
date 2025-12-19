# Data Directory

This directory contains the training, development, and test data used for the emotion classification project.

## Files

- `presidential_speeches_goemotions_labeled.csv` - Presidential speeches dataset labeled with GoEmotions emotions

## Data Format

The labeled dataset contains the following columns:
- `speech` / `text` / `content` - The text of the speech
- `primary_emotion` - The primary emotion label (one of 28 GoEmotions categories)
- Additional metadata columns

## GoEmotions Labels (28 categories)

```
admiration, amusement, anger, annoyance, approval, caring,
confusion, curiosity, desire, disappointment, disapproval,
disgust, embarrassment, excitement, fear, gratitude, grief,
joy, love, nervousness, optimism, pride, realization,
relief, remorse, sadness, surprise, neutral
```

## How to Regenerate Labels

If you need to regenerate the labeled dataset:

```bash
cd code/
jupyter notebook label_presidential.ipynb
# Execute all cells
```

This will create `presidential_speeches_goemotions_labeled.csv` using the pre-trained GoEmotions model.

## Original Data Source

The original presidential speeches data can be downloaded from:
- The `retrieve_data.ipynb` notebook in the `code/` directory downloads data from Google's GoEmotions dataset
- Presidential speeches metadata is included in `1presidential_speeches_with_metadata.xlsx`

## Note on Large Data

The GoEmotions training data is loaded directly from Hugging Face during model training:
```python
from datasets import load_dataset
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
```
