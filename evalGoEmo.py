"""
Evaluation Script for GoEmotions Test Set

This script evaluates trained models on the official GoEmotions test set.
It automatically detects model types (BERT, RoBERTa, DeBERTa, ModernBERT) based on filenames.

Usage:
    python evalGoEmo.py
"""

import os
import glob
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    BertTokenizer, 
    BertForSequenceClassification,
    DebertaV2Tokenizer
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

# GoEmotions Labels (28 labels)
LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
NUM_LABELS = len(LABELS)

class GoEmotionsDataset(Dataset):
    def __init__(self, split="test", tokenizer=None, max_length=128):
        self.dataset = load_dataset("google-research-datasets/go_emotions", "simplified")[split]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label_ids = item['labels']
        
        # Create multi-hot vector
        labels = torch.zeros(NUM_LABELS)
        labels[label_ids] = 1.0
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

def get_model_and_tokenizer(model_path, device):
    """Load the appropriate model and tokenizer based on filename."""
    model_path_lower = model_path.lower()
    
    if "deberta" in model_path_lower:
        base_model = "microsoft/deberta-v3-base"
        print(f"[INFO] Detected DeBERTa model (base: {base_model})")
        
        tokenizer = DebertaV2Tokenizer.from_pretrained(base_model)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification"
        )
    elif "roberta" in model_path_lower:
        base_model = "roberta-base"
        print(f"[INFO] Detected RoBERTa model (base: {base_model})")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification"
        )
    elif "modernbert" in model_path_lower:
        base_model = "answerdotai/ModernBERT-base"
        print(f"[INFO] Detected ModernBERT model (base: {base_model})")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification"
        )
    else:
        # Default to BERT
        base_model = "bert-base-uncased"
        print(f"[INFO] Detected BERT model (base: {base_model})")
        tokenizer = BertTokenizer.from_pretrained(base_model)
        model = BertForSequenceClassification.from_pretrained(
            base_model,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification"
        )
    
    # Load trained weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"[ERROR] Failed to load state dict for {model_path}: {e}")
        return None, None

    model.to(device)
    model.eval()
    return model, tokenizer

def evaluate_model(model, tokenizer, device, batch_size=32):
    """Evaluate a single model on GoEmotions test set."""
    test_dataset = GoEmotionsDataset(split="test", tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds) # Exact match accuracy
    
    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'accuracy': accuracy
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find all .pt files excluding 'trial'
    model_files = [f for f in glob.glob("*.pt") if "trial" not in f.lower()]
    
    if not model_files:
        print("No model files found!")
        return

    print(f"Found {len(model_files)} models to evaluate: {model_files}")
    
    results = []

    # Evaluate Pretrained RoBERTa Model
    pretrained_model_name = "SamLowe/roberta-base-go_emotions"
    print(f"\n{'='*60}")
    print(f"Evaluating Pretrained: {pretrained_model_name}")
    print(f"{'='*60}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)
        model.to(device)
        model.eval()
        
        metrics = evaluate_model(model, tokenizer, device)
        print(f"Macro F1: {metrics['f1_macro']:.4f}")
        print(f"Micro F1: {metrics['f1_micro']:.4f}")
        
        results.append({
            'Model': pretrained_model_name,
            'Macro F1': metrics['f1_macro'],
            'Micro F1': metrics['f1_micro']
        })
    except Exception as e:
        print(f"[ERROR] Failed to evaluate {pretrained_model_name}: {e}")
    
    for model_path in model_files:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_path}")
        print(f"{'='*60}")
        
        model, tokenizer = get_model_and_tokenizer(model_path, device)
        if model is None:
            continue
            
        metrics = evaluate_model(model, tokenizer, device)
        
        print(f"Macro F1: {metrics['f1_macro']:.4f}")
        print(f"Micro F1: {metrics['f1_micro']:.4f}")
        
        results.append({
            'Model': model_path,
            'Macro F1': metrics['f1_macro'],
            'Micro F1': metrics['f1_micro']
        })
    
    # Print Summary Table
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY (GoEmotions Test Set)")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values(by='Macro F1', ascending=False)
        print(df_results.to_string(index=False))
        
        best_model = df_results.iloc[0]
        print(f"\n🏆 Best Model: {best_model['Model']} (Macro F1: {best_model['Macro F1']:.4f})")
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()
