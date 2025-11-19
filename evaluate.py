import sys
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_labels(filename):
    """
    Reads labels from a text file, assuming one label per line.
    Strips whitespace and converts to uppercase for consistency.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        # Read lines and strip whitespace
        labels = [line.strip().upper() for line in f]
    return labels

def main():
    parser = argparse.ArgumentParser(description='Evaluate Emotion Classification System.')
    parser.add_argument('gold_file', help='Path to the Gold Standard (correct) answers file.')
    parser.add_argument('sys_file', help='Path to the System Output (predicted) file.')
    args = parser.parse_args()

    # Load the data
    try:
        gold_labels = load_labels(args.gold_file)
        sys_labels = load_labels(args.sys_file)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)

    # Sanity check
    if len(gold_labels) != len(sys_labels):
        print(f"Error: Mismatch in line counts. Gold: {len(gold_labels)}, System: {len(sys_labels)}")
        sys.exit(1)

    # Calculate Metrics
    # Macro average is best for emotion classification (treats all classes equally)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_labels, 
        sys_labels, 
        average='macro', 
        zero_division=0
    )
    
    accuracy = accuracy_score(gold_labels, sys_labels)

    # Output results
    print("-" * 30)
    print("EVALUATION RESULTS")
    print("-" * 30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print("-" * 30)
    
    # This single number is your "Score" for the project
    print(f"FINAL SCORE (Macro F1): {f1:.4f}")

if __name__ == "__main__":
    main()