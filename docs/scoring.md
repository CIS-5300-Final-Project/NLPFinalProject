How to run the script:

**python evaluate.py gold_answers.txt system_predictions.txt**


For this project, the primary evaluation metric is Macro-averaged F1 score. This metric is chosen over accuracy to account for class imbalance, which is typical in emotion classification tasks (ex. "Joy" is more common in the dataset than "Disgust").


The F1 score is the harmonic mean of the Precision and Recall. Macro-averaging computes the F1 score independetly for each class (emotion), and takes the unweighted mean of these scores. This ensures that the performance on minority classes (rare emotions) contributes equally to the final score as majority classes.


Paper Reference: 

This metric was the official evaluation metric for the  **SemEval-2018 Task 1: Affect in Tweets** specifically the emotion classification subtask.

* *Mohammad, S., Bravo-Marquez, F., Salameh, M., & Kiritchenko, S. (2018). SemEval-2018 Task 1: Affect in Tweets. Proceedings of the 12th International Workshop on Semantic Evaluation.* [Link to Paper](https://aclanthology.org/S18-1001/)

Wikipedia Definitions:

* [F-score](https://en.wikipedia.org/wiki/F-score)
* [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
