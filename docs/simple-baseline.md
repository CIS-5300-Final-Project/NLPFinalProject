# Simple Baseline: Majority-Class Emotion Classifier

## Overview
This document describes the simple baseline model used for our political speech **emotion classification** task.  
The goal of this baseline is to set a minimal performance benchmark that any meaningful model should exceed.

Our classification task assumes four possible emotion categories:

- **angry**
- **fearful**
- **neutral**
- **....**

The baseline classifier uses a **majority class strategy**:  
it selects the most frequent emotion in the training set and predicts this emotion for *all* test examples.

---

## Baseline Method

### Majority-Class Model
Given training data `train_df`, the baseline:

1. Checks whether the dataset contains an `emotion` column. 
2. Computes the majority emotion:  
   ```python
   self.majority_class = train_df["emotion"].value_counts().idxmax()
