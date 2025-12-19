# NLPFinalProject

Models: The models are too big in size, so they are included in the google drive folder found here:
[https://drive.google.com/drive/folders/1cAxlNEXwFEmKOICnVLyKcmT-ZgmZA4ZX?usp=sharing]()

## Run eval script:

```
python scripts/evaluate.py models/best_bert_model.pt
```

The eval script compares our majority baseline model, to the goemotions trained model from the research paper, to our own trained bert model

## Labelling our dataset:

Run `notebooks/label_presidential.ipynb` to label and save our dataset with pretrained go emotions model

## Results:

![1766102526426](image/README/1766102526426.png)

## Steps to run the code:

1. Run `notebooks/label_presidential.ipynb` to label the presidential dataset
2. Run `notebooks/bert.ipynb` to train the custom bert model
3. Run `python scripts/evaluate.py models/best_bert_model.pt` to evaluate the models against the presidential dataset
