# Code Directory

This directory contains all code developed for the NLP Final Project on emotion classification using BERT-based models.

## Directory Structure

```
code/
├── bert.ipynb                  # BERT model training notebook
├── deberta.ipynb               # DeBERTa model training notebook
├── modernbert.ipynb            # ModernBERT model training notebook
├── roberta.ipynb               # RoBERTa model training notebook
├── label_presidential.ipynb    # Labels presidential speeches with GoEmotions
├── load_data.ipynb             # Data loading utilities notebook
├── retrieve_data.ipynb         # Data retrieval notebook
├── evaluate.py                 # Main evaluation script
├── evalGoEmo.py                # GoEmotions test set evaluation
├── simple-baseline.py          # Simple majority-class baseline
├── scoring.md                  # Scoring documentation
├── simple-baseline.md          # Baseline documentation
├── bert-baseline/              # BERT baseline implementation
├── llama_model/                # LLaMA model experiments
└── image/                      # Images for documentation
```

## Step-by-Step Instructions

### Prerequisites

```bash
pip install torch transformers datasets pandas numpy scikit-learn tqdm
```

### Step 1: Label the Presidential Dataset

Run the labeling notebook to create emotion labels for the presidential speeches:

```bash
# From the code/ directory
jupyter notebook label_presidential.ipynb
# Execute all cells to generate: ../data/presidential_speeches_goemotions_labeled.csv
```

### Step 2: Train Models

Train any of the BERT-based models. Example for BERT:

```bash
# From the code/ directory
jupyter notebook bert.ipynb
# Execute all cells - model will be saved to ../output/best_bert_model.pt
```

Available model notebooks:
- `bert.ipynb` → `best_bert_model.pt`
- `roberta.ipynb` → `best_roberta_model.pt`
- `deberta.ipynb` → `best_deberta_model.pt`
- `modernbert.ipynb` → `best_modernbert_model.pt`

### Step 3: Evaluate Models

#### Evaluate on Presidential Speeches Dataset

```bash
# From the code/ directory
python evaluate.py ../output/best_bert_model.pt
```

To evaluate multiple models:
```bash
python evaluate.py ../output/best_bert_model.pt ../output/best_roberta_model.pt
```

Options:
- `--data PATH`: Custom path to labeled data (default: `../data/presidential_speeches_goemotions_labeled.csv`)
- `--output-dir PATH`: Output directory for results (default: `../output`)
- `--skip-roberta`: Skip the RoBERTa baseline evaluation

#### Evaluate on GoEmotions Test Set

```bash
# From the code/ directory
python evalGoEmo.py
```

This will evaluate all models in `../output/` on the official GoEmotions test set.

### Example Command Lines to Reproduce Results

```bash
cd code/

# 1. Evaluate BERT model on presidential speeches
python evaluate.py ../output/best_presidential_bert_model.pt

# 2. Evaluate RoBERTa model on presidential speeches
python evaluate.py ../output/best_presidential_roberta_model.pt

# 3. Evaluate all models on GoEmotions test set
python evalGoEmo.py

# 4. Run simple baseline
python simple-baseline.py --data ../data/presidential_speeches_goemotions_labeled.csv
```

## Output

The evaluation scripts output:
- Accuracy and F1 scores for each model
- Comparison tables
- Error analysis data saved to `../output/error_analysis_data.txt`
