import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input='all_text.txt',
    model_prefix='bert_sp',
    vocab_size=8000,
    character_coverage=1.0,
    model_type='bpe'
)
