import datetime
import random
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from absl import app, flags, logging
from loguru import logger
from scipy import stats
from sklearn import metrics, model_selection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
import config
import dataset
import engine
from model import BERTBaseUncased

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

writer = SummaryWriter()
logger.add("experiment.log")

flags.DEFINE_boolean('features', True, "")
flags.DEFINE_string('test_file', None, "")
flags.DEFINE_string('model_path', None, "")

FLAGS = flags.FLAGS

def main(_):
    test_file = config.EVAL_PROC
    model_path = config.MODEL_PATH
    if FLAGS.test_file:
        test_file = FLAGS.test_file
    if FLAGS.model_path:
        model_path = FLAGS.model_path
    df_test = pd.read_fwf(test_file)

    logger.info(f"Bert Model: {config.BERT_PATH}")
    logger.info(
        f"Current date and time :{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
    logger.info(f"Test file: {test_file}")
    logger.info(f"Test size : {len(df_test):.4f}")
    
    trg = []
    for i in range(len(df_test.values)):
        trg.append(0)

    test_dataset = dataset.BERTDataset(
        text=df_test.values,
        target=trg
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=3
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BERTBaseUncased(config.DROPOUT)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)

    outputs, extracted_features = engine.predict_fn(
        test_data_loader, model, device, extract_features=FLAGS.features)
    df_test["predicted"] = outputs
    # save file
    df_test.to_csv('/home/matiss/experiments/embeddings/lm/tune-bert-2021/fulltext.prf.predictions-best.csv', header=None, index=False)


if __name__ == "__main__":
    app.run(main)
