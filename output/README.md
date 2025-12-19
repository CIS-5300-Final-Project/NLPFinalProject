# Output Directory

This directory contains model predictions, trained models, and evaluation results.

## Trained Models (.pt files)

### GoEmotions Trained Models

- `best_bert_model.pt` - BERT model trained on GoEmotions
- `best_roberta_model.pt` - RoBERTa model trained on GoEmotions
- `best_deberta_model.pt` - DeBERTa model trained on GoEmotions
- `best_modernbert_model.pt` - ModernBERT model trained on GoEmotions

### Presidential Speeches Fine-tuned Models

- `best_presidential_bert_model.pt` - BERT fine-tuned on presidential speeches
- `best_presidential_roberta_model.pt` - RoBERTa fine-tuned on presidential speeches
- `best_presidential_deberta_model.pt` - DeBERTa fine-tuned on presidential speeches
- `best_presidential_modernbert_model.pt` - ModernBERT fine-tuned on presidential speeches

### Hyperparameter Tuning Models

- `trial_0_best.pt` through `trial_9_best.pt` - Models from hyperparameter tuning trials
- `best_tuned_bert_model.pt` - Best model from hyperparameter tuning

## Evaluation Results

- `model_comparison.csv` - Comparison of all models' performance
- `per_class_comparison.csv` - Per-class metrics comparison
- `error_analysis_data.txt` - Detailed error analysis between baseline and best model
- `hyperparameter_tuning_results.json` - Results from hyperparameter tuning

## Visualizations

- `per_class_comparison.png` - Per-class performance visualization
- `hyperparameter_tuning_results.png` - Hyperparameter tuning results chart

## Running Evaluation

To run the evaluation script and generate new results:

```bash
cd code/
python evaluate.py ../output/best_presidential_bert_model.pt
```

### Example Output

```
============================================================
Model                                       Acc       F1
------------------------------------------------------------
best_presidential_bert_model.pt          0.4521   0.2847
SamLowe/RoBERTa-base-go_emotions         0.3892   0.2156
Simple Baseline (Majority Class)          0.1876   0.0312
============================================================
```

### Evaluation Script Options

```bash
python evaluate.py [model_files] [options]

Options:
  --data PATH         Path to labeled dataset (default: ../data/presidential_speeches_goemotions_labeled.csv)
  --output-dir PATH   Output directory for results (default: ../output)
  --skip-roberta      Skip RoBERTa baseline evaluation
```

### Evaluating on GoEmotions Test Set

```bash
cd code/
python evalGoEmo.py
```

This evaluates all models in this directory on the official GoEmotions test set.
