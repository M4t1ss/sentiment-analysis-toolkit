import transformers

MAX_LEN = 256 #256
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-5
DROPOUT = 0.3
SAVE = 15000

#Load model for tuning or prediction or save model while training/tuning
MODEL_PATH = "/home/matiss/tools/SentimentAnalyserLVTwitter/sentiment-models/twitediens_2021_grid_model.74.06.bin"

# Processed training, development and evaluation files
TRAIN_PROC = "/home/matiss/experiments/embeddings/emo/data/other/only-url-usr/full-auto-clean.csv"
DEVEL_PROC = "/home/matiss/experiments/embeddings/emo/data/other/only-url-usr/xmp-eval.csv"
EVAL_PROC = "/home/matiss/experiments/embeddings/emo/data/other/only-url-usr/xtwitediens.test.csv"

# MBERT Raw Version
# BERT_PATH = "bert-base-multilingual-cased"

BERT_PATH = "/home/matiss/tools/SentimentAnalyserLVTwitter/bert-models/rinalds/tuned-further"

# BertTokenizer
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=False
)
