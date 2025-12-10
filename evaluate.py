"""
Evaluation Script for Ekman Emotion Classification

This script evaluates multiple models on the presidential speeches dataset:
1. RoBERTa-base-go_emotions (pre-trained baseline)
2. Simple majority-class baseline
3. Custom .pt models (BERT-based)

Usage:
    python evaluate.py model1.pt model2.pt ...
    python evaluate.py --help
"""

import sys
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# Ekman emotion labels
EKMAN_LABELS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

# Mapping from GoEmotions (28 labels) to Ekman (7 labels) for RoBERTa model
GOEMOTIONS_TO_EKMAN = {
    'anger': 'anger',
    'annoyance': 'anger',
    'disapproval': 'anger',
    'disgust': 'disgust',
    'fear': 'fear',
    'nervousness': 'fear',
    'joy': 'joy',
    'amusement': 'joy',
    'approval': 'joy',
    'excitement': 'joy',
    'gratitude': 'joy',
    'love': 'joy',
    'optimism': 'joy',
    'relief': 'joy',
    'pride': 'joy',
    'admiration': 'joy',
    'desire': 'joy',
    'caring': 'joy',
    'sadness': 'sadness',
    'disappointment': 'sadness',
    'embarrassment': 'sadness',
    'grief': 'sadness',
    'remorse': 'sadness',
    'surprise': 'surprise',
    'realization': 'surprise',
    'confusion': 'surprise',
    'curiosity': 'surprise',
    'neutral': 'neutral',
}


def load_presidential_data(path="data/presidential_speeches_ekman_labeled.csv"):
    """Load the presidential speeches dataset with Ekman labels."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} presidential speeches")
    
    # Check if required columns exist
    if 'primary_emotion' not in df.columns:
        raise ValueError("Dataset must have 'primary_emotion' column. Run label_presendential.ipynb first.")
    
    # Find text column
    text_col = None
    for col in ['speech', 'text', 'content']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError("Could not find text column (expected 'speech', 'text', or 'content')")
    
    print(f"[INFO] Using text column: '{text_col}'")
    print(f"[INFO] Ground truth label distribution:")
    print(df['primary_emotion'].value_counts())
    
    return df, text_col


def evaluate_simple_baseline(df):
    """
    Evaluate simple majority-class baseline.
    Always predicts the most common class.
    """
    gold_labels = df['primary_emotion'].str.lower().tolist()
    
    # Find majority class
    majority_class = df['primary_emotion'].value_counts().idxmax().lower()
    print(f"[INFO] Majority class: {majority_class}")
    
    # Predict majority class for all
    predictions = [majority_class] * len(gold_labels)
    
    return gold_labels, predictions, "Simple Baseline (Majority Class)"


def evaluate_roberta_goemotions(df, text_col, device, batch_size=8):
    """
    Evaluate the SamLowe/roberta-base-go_emotions model.
    Maps 28 GoEmotions to 7 Ekman emotions.
    """
    print("\n[INFO] Loading RoBERTa-base-go_emotions model...")
    
    model_name = "SamLowe/roberta-base-go_emotions"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Get label mapping from model config
    id2label = model.config.id2label
    
    gold_labels = []
    predictions = []
    
    print("[INFO] Running inference...")
    for idx in tqdm(range(len(df)), desc="RoBERTa Inference"):
        text = str(df.iloc[idx][text_col])[:5000]  # Truncate
        gold = df.iloc[idx]['primary_emotion'].lower()
        gold_labels.append(gold)
        
        # Tokenize
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        
        # Aggregate to Ekman emotions
        ekman_probs = {label: 0.0 for label in EKMAN_LABELS}
        for i, prob in enumerate(probs):
            goemotions_label = id2label[i]
            if goemotions_label in GOEMOTIONS_TO_EKMAN:
                ekman_label = GOEMOTIONS_TO_EKMAN[goemotions_label]
                ekman_probs[ekman_label] = max(ekman_probs[ekman_label], prob)
        
        # Predict the emotion with highest probability
        pred = max(ekman_probs, key=ekman_probs.get)
        predictions.append(pred)
    
    return gold_labels, predictions, "RoBERTa-base-go_emotions"


def evaluate_bert_model(df, text_col, model_path, device, batch_size=8):
    """
    Evaluate a custom BERT .pt model.
    Assumes the model is a BertForSequenceClassification with 7 Ekman labels.
    """
    print(f"\n[INFO] Loading BERT model from {model_path}...")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(EKMAN_LABELS),
        problem_type="multi_label_classification"
    )
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    gold_labels = []
    predictions = []
    
    print("[INFO] Running inference...")
    for idx in tqdm(range(len(df)), desc=f"BERT Inference ({model_path})"):
        text = str(df.iloc[idx][text_col])[:5000]  # Truncate
        gold = df.iloc[idx]['primary_emotion'].lower()
        gold_labels.append(gold)
        
        # Tokenize
        inputs = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        
        # Map to emotion labels and predict highest
        label_probs = {label: prob for label, prob in zip(EKMAN_LABELS, probs)}
        pred = max(label_probs, key=label_probs.get)
        predictions.append(pred)
    
    model_name = model_path.split('/')[-1].split('\\')[-1]
    return gold_labels, predictions, f"BERT Model ({model_name})"


def compute_metrics(gold_labels, predictions, model_name):
    """Compute and display evaluation metrics."""
    accuracy = accuracy_score(gold_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_labels, predictions, average='macro', zero_division=0
    )
    
    results = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1
    }
    
    return results


def print_results(all_results):
    """Print comparison table of all models."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS COMPARISON")
    print("=" * 80)
    
    # Header
    print(f"{'Model':<45} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 (Macro)':>12}")
    print("-" * 80)
    
    # Sort by F1 score descending
    sorted_results = sorted(all_results, key=lambda x: x['f1_macro'], reverse=True)
    
    for r in sorted_results:
        print(f"{r['model']:<45} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1_macro']:>12.4f}")
    
    print("-" * 80)
    
    # Best model
    best = sorted_results[0]
    print(f"\n🏆 BEST MODEL: {best['model']}")
    print(f"   Macro F1 Score: {best['f1_macro']:.4f}")
    print(f"   Accuracy: {best['accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate emotion classification models on presidential speeches dataset.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate.py                          # Evaluate baselines only
    python evaluate.py best_bert_model.pt       # Evaluate baselines + custom model
    python evaluate.py model1.pt model2.pt      # Evaluate multiple models
        """
    )
    parser.add_argument(
        'model_files', 
        nargs='*', 
        help='Paths to .pt model files to evaluate (optional)'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='data/presidential_speeches_ekman_labeled.csv',
        help='Path to the labeled presidential speeches CSV'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=8,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--skip-roberta', 
        action='store_true',
        help='Skip RoBERTa-base-go_emotions evaluation'
    )
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    try:
        df, text_col = load_presidential_data(args.data)
    except FileNotFoundError:
        print(f"[ERROR] Data file not found: {args.data}")
        print("[INFO] Please ensure you have run the labeling notebook first.")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    all_results = []
    
    # 1. Evaluate simple baseline
    print("\n" + "=" * 50)
    print("Evaluating: Simple Majority-Class Baseline")
    print("=" * 50)
    gold, preds, name = evaluate_simple_baseline(df)
    results = compute_metrics(gold, preds, name)
    all_results.append(results)
    print(f"Accuracy: {results['accuracy']:.4f}, F1 (Macro): {results['f1_macro']:.4f}")
    
    # 2. Evaluate RoBERTa-base-go_emotions
    if not args.skip_roberta:
        print("\n" + "=" * 50)
        print("Evaluating: RoBERTa-base-go_emotions")
        print("=" * 50)
        try:
            gold, preds, name = evaluate_roberta_goemotions(df, text_col, device, args.batch_size)
            results = compute_metrics(gold, preds, name)
            all_results.append(results)
            print(f"Accuracy: {results['accuracy']:.4f}, F1 (Macro): {results['f1_macro']:.4f}")
        except Exception as e:
            print(f"[WARNING] Failed to evaluate RoBERTa model: {e}")
    
    # 3. Evaluate custom .pt models
    for model_path in args.model_files:
        print("\n" + "=" * 50)
        print(f"Evaluating: {model_path}")
        print("=" * 50)
        try:
            gold, preds, name = evaluate_bert_model(df, text_col, model_path, device, args.batch_size)
            results = compute_metrics(gold, preds, name)
            all_results.append(results)
            print(f"Accuracy: {results['accuracy']:.4f}, F1 (Macro): {results['f1_macro']:.4f}")
        except FileNotFoundError:
            print(f"[ERROR] Model file not found: {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {model_path}: {e}")
    
    # Print comparison
    print_results(all_results)


if __name__ == "__main__":
    main()