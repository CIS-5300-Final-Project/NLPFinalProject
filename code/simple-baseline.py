import argparse
from load_data import *

class BaselineEmotionClassifier:
    def __init__(self):
        self.majority_class = None

    def train(self, train_df):
        if "emotion" not in train_df.columns:
            print("'emotion' column not found. Creating one filled with 'neutral'.")
            train_df["emotion"] = "neutral"

        self.majority_class = train_df["emotion"].value_counts().idxmax()
        print(f"Majority emotion = '{self.majority_class}'")

    def predict(self, df):
        if self.majority_class is None:
            raise RuntimeError("Model not trained yet. Call train() first.")

        preds = df.copy()
        preds["predicted_emotion"] = self.majority_class
        return preds

    def save_predictions(self, predictions_df, output_path):
        predictions_df[["predicted_emotion"]].to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/1presidential_speeches_with_metadata.xlsx")
    parser.add_argument("--output", type=str, default="../output/baseline_predictions.csv")
    args = parser.parse_args()

    train_df, val_df, test_df = load_and_split_data(args.data)

    baseline = BaselineEmotionClassifier()
    baseline.train(train_df)

    predictions = baseline.predict(test_df)

    baseline.save_predictions(predictions, args.output)

    print("Simple baseline completed.")


if __name__ == "__main__":
    main()