# NLPFinalProject

Run eval script:
**python evaluate.py best_bert_model.pt**

The eval script compares our majority baseline model, to the goemotions trained model from the research paper, to our own trained bert model

Labelling our dataset:

run the label_presidential.ipynb to label and save our dataset with pretrained go emotions model

Results:

![1765408866241](image/README/1765408866241.png)


Steps to run the code:

1. run label_presidential.ipynb to label the presidential dataset
2. run bert_ipynb to train the custom bert model
3. run "python evaluate.py best_bert_model.pt" to evaluate the models against the presidential dataset
