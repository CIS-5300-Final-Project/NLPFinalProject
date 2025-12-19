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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from collections import Counter

# All 28 GoEmotions labels
GOEMOTIONS_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def load_presidential_data(path="data/presidential_speeches_goemotions_labeled.csv"):
    """Load the presidential speeches dataset with GoEmotions labels."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Could not read file {path}: {e}")
        sys.exit(1)
        
    print(f"[INFO] Loaded {len(df)} presidential speeches")
    
    if 'primary_emotion' not in df.columns:
        raise ValueError("Dataset must have 'primary_emotion' column.")
    
    text_col = None
    for col in ['speech', 'text', 'content']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError("Could not find text column (expected 'speech', 'text', or 'content')")
    
    return df, text_col

def evaluate_simple_baseline(df):
    gold_labels = df['primary_emotion'].str.lower().tolist()
    majority_class = df['primary_emotion'].value_counts().idxmax().lower()
    predictions = [majority_class] * len(gold_labels)
    return gold_labels, predictions, "Simple Baseline (Majority Class)"

def evaluate_roberta_goemotions(df, text_col, device, batch_size=8):
    print("\n[INFO] Loading RoBERTa-base-go_emotions model...")
    model_name = "SamLowe/roberta-base-go_emotions"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    id2label = model.config.id2label
    gold_labels = []
    predictions = []
    
    for idx in tqdm(range(len(df)), desc="RoBERTa Inference"):
        text = str(df.iloc[idx][text_col])[:2000] # Truncate for speed/memory
        gold = df.iloc[idx]['primary_emotion'].lower()
        gold_labels.append(gold)
        
        inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        
        label_probs = {id2label[i]: prob for i, prob in enumerate(probs)}
        pred = max(label_probs, key=label_probs.get)
        predictions.append(pred)
    
    # --- FIX: Ensure the return name matches what generate_error_report looks for ---
    return gold_labels, predictions, "SamLowe/RoBERTa-base-go_emotions"

def evaluate_custom_model(df, text_col, model_path, device):
    print(f"\n[INFO] Loading model from {model_path}...")
    
    if "deberta" in model_path.lower():
        base_model = "microsoft/deberta-v3-base"
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=28, problem_type="multi_label_classification")
    elif "roberta" in model_path.lower():
        base_model = "roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=28, problem_type="multi_label_classification")
    elif "modernbert" in model_path.lower():
        base_model = "answerdotai/ModernBERT-base"
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=28, problem_type="multi_label_classification")
    else:
        base_model = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(base_model)
        model = BertForSequenceClassification.from_pretrained(base_model, num_labels=28, problem_type="multi_label_classification")
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        return [], [], "ERROR"

    model.to(device)
    model.eval()
    
    gold_labels = []
    predictions = []
    
    for idx in tqdm(range(len(df)), desc=f"Inference ({model_path})"):
        text = str(df.iloc[idx][text_col])[:2000]
        gold = df.iloc[idx]['primary_emotion'].lower()
        gold_labels.append(gold)
        
        inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        
        # Map 0-27 indices to labels
        label_probs = {label: prob for label, prob in zip(GOEMOTIONS_LABELS, probs)}
        pred = max(label_probs, key=label_probs.get)
        predictions.append(pred)
    
    model_name = model_path.split('/')[-1].split('\\')[-1]
    return gold_labels, predictions, model_name

def compute_metrics(gold_labels, predictions, model_name):
    if not predictions: return {'model': model_name, 'accuracy': 0, 'f1_macro': 0}
    
    accuracy = accuracy_score(gold_labels, predictions)
    _, _, f1, _ = precision_recall_fscore_support(gold_labels, predictions, average='macro', zero_division=0)
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'f1_macro': f1,
        'predictions': predictions # Store for error analysis
    }

def generate_error_report(df, text_col, all_results, filename="error_analysis_data.txt"):
    """
    Generates a text file comparing the Best Custom Model vs. Published Baseline.
    """
    print(f"\n[INFO] Generating error analysis report to {filename}...")
    
    # 1. Identify Baseline (Look for "SamLowe" in the name)
    baseline_result = next((r for r in all_results if "SamLowe" in r['model']), None)
    
    # 2. Identify Custom Models (Exclude SamLowe and Simple Baseline)
    custom_results = [r for r in all_results if "SamLowe" not in r['model'] and "Simple Baseline" not in r['model']]
    
    if not custom_results:
        print("[WARNING] No custom models found to compare. Skipping detailed report.")
        return
        
    if not baseline_result:
        print("[WARNING] Baseline (SamLowe) not found. Skipping detailed report.")
        return

    best_model_result = max(custom_results, key=lambda x: x['f1_macro'])
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== ERROR ANALYSIS RAW DATA ===\n")
        f.write(f"Baseline Model: {baseline_result['model']}\n")
        f.write(f"Best Custom Model: {best_model_result['model']}\n")
        f.write("===============================\n\n")

        # Get data
        gold = df['primary_emotion'].str.lower().tolist()
        texts = df[text_col].tolist()
        base_preds = baseline_result['predictions']
        best_preds = best_model_result['predictions']
        
        # --- SECTION 1: CONFUSION MATRIX (Top Errors) ---
        f.write("--- TOP 10 CONFUSED PAIRS (Best Model) ---\n")
        f.write("Format: (True Label) -> (Predicted Label): Count\n")
        
        pairs = list(zip(gold, best_preds))
        error_pairs = [p for p in pairs if p[0] != p[1]]
        cnt = Counter(error_pairs)
        
        for (true_l, pred_l), count in cnt.most_common(10):
            f.write(f"True: [{true_l}] -> Pred: [{pred_l}] : {count} times\n")
        f.write("\n")

        # --- SECTION 2: BASELINE WRONG -> CUSTOM CORRECT (Improvements) ---
        f.write("--- IMPROVEMENTS (Baseline Wrong -> Custom Correct) ---\n")
        count = 0
        for i in range(len(gold)):
            if count >= 5: break # Limit examples
            if base_preds[i] != gold[i] and best_preds[i] == gold[i]:
                f.write(f"Example {count+1}:\n")
                f.write(f"Text: {texts[i][:200]}...\n")
                f.write(f"Gold: {gold[i]}\n")
                f.write(f"Baseline Pred: {base_preds[i]} (WRONG)\n")
                f.write(f"Custom Pred:   {best_preds[i]} (CORRECT)\n")
                f.write("-" * 20 + "\n")
                count += 1
        f.write("\n")

        # --- SECTION 3: BASELINE CORRECT -> CUSTOM WRONG (Regressions) ---
        f.write("--- REGRESSIONS (Baseline Correct -> Custom Wrong) ---\n")
        count = 0
        for i in range(len(gold)):
            if count >= 5: break 
            if base_preds[i] == gold[i] and best_preds[i] != gold[i]:
                f.write(f"Example {count+1}:\n")
                f.write(f"Text: {texts[i][:200]}...\n")
                f.write(f"Gold: {gold[i]}\n")
                f.write(f"Baseline Pred: {base_preds[i]} (CORRECT)\n")
                f.write(f"Custom Pred:   {best_preds[i]} (WRONG)\n")
                f.write("-" * 20 + "\n")
                count += 1
        
        # --- SECTION 4: BOTH WRONG ---
        f.write("\n--- PERSISTENT ERRORS (Both Wrong) ---\n")
        count = 0
        for i in range(len(gold)):
            if count >= 5: break
            if base_preds[i] != gold[i] and best_preds[i] != gold[i]:
                f.write(f"Example {count+1}:\n")
                f.write(f"Text: {texts[i][:200]}...\n")
                f.write(f"Gold: {gold[i]}\n")
                f.write(f"Baseline Pred: {base_preds[i]}\n")
                f.write(f"Custom Pred:   {best_preds[i]}\n")
                f.write("-" * 20 + "\n")
                count += 1

    print(f"[SUCCESS] Report written to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate emotion classification models.')
    parser.add_argument('model_files', nargs='*', help='Paths to .pt model files')
    parser.add_argument('--data', type=str, default='data/presidential_speeches_goemotions_labeled.csv')
    parser.add_argument('--skip-roberta', action='store_true', help='Skip RoBERTa baseline')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    try:
        df, text_col = load_presidential_data(args.data)
    except Exception as e:
        print(e)
        return

    all_results = []
    
    # 1. Simple Baseline
    g, p, n = evaluate_simple_baseline(df)
    all_results.append(compute_metrics(g, p, n))

    # 2. RoBERTa Baseline (SamLowe)
    if not args.skip_roberta:
        g, p, n = evaluate_roberta_goemotions(df, text_col, device)
        all_results.append(compute_metrics(g, p, n))
        
    # 3. Custom Models
    for model_path in args.model_files:
        g, p, n = evaluate_custom_model(df, text_col, model_path, device)
        all_results.append(compute_metrics(g, p, n))

    # Print Summary Table
    print("\n" + "="*60)
    print(f"{'Model':<40} {'Acc':>8} {'F1':>8}")
    print("-" * 60)
    for r in sorted(all_results, key=lambda x: x['f1_macro'], reverse=True):
        print(f"{r['model']:<40} {r['accuracy']:>8.4f} {r['f1_macro']:>8.4f}")
    print("="*60)

    # Generate Error Report
    generate_error_report(df, text_col, all_results)

if __name__ == "__main__":
    main()