import random
import datetime
import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import os,sys
from absl import app, flags
from model import BERTBaseCased
from sklearn import model_selection, metrics
from sklearn.utils import shuffle
from transformers import AdamW, get_linear_schedule_with_warmup
from loguru import logger
from torch.utils.tensorboard import SummaryWriter


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

writer = SummaryWriter()
logger.add("experiment.log")

flags.DEFINE_float('lr', 0.00001, "")
flags.DEFINE_float('dropout', 0.3, "")
flags.DEFINE_float('save', 10000, "")
flags.DEFINE_bool('tune', False, "")
flags.DEFINE_integer('estop', 5, "")

FLAGS = flags.FLAGS

def main(_):
    LEARNING_RATE = config.LEARNING_RATE
    DROPOUT = config.DROPOUT
    SAVE = config.SAVE
    TUNE = False
    ESTOP = 5

    if FLAGS.lr:
        LEARNING_RATE = FLAGS.lr
    if FLAGS.dropout:
        DROPOUT = FLAGS.dropout
    if FLAGS.save:
        SAVE = FLAGS.save
    if FLAGS.tune:
        TUNE = FLAGS.tune
    if FLAGS.estop:
        ESTOP = FLAGS.estop

    train_file = config.TRAIN_PROC
    df_train = pd.read_csv(train_file).fillna("none")

    valid_file = config.DEVEL_PROC
    df_valid = pd.read_csv(valid_file).fillna("none")

    test_file = config.EVAL_PROC
    df_test = pd.read_csv(test_file).fillna("none")
    
    logger.info(f"Bert Model: {config.BERT_PATH}")
    logger.info(f"Current date and time :{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")

    logger.info(f"Train file: {train_file}")
    logger.info(f"Valid file: {valid_file}")
    logger.info(f"Test file: {test_file}")

    logger.info(f"Train size : {len(df_train):.4f}")
    logger.info(f"Valid size : {len(df_valid):.4f}")
    logger.info(f"Test size : {len(df_test):.4f}")
    

    valid_dataset = dataset.BERTDataset(
        text=df_valid.text.values,
        target=df_valid.label.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    test_dataset = dataset.BERTDataset(
        text=df_test.text.values,
        target=df_test.label.values
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #torch.device("cuda")
    model = BERTBaseCased(DROPOUT)
    if TUNE:
        model.load_state_dict(torch.load(configtune.MODEL_PATH, map_location=torch.device(device)))
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # model = nn.DataParallel(model)

    best_accuracy = 0
    best_path = ""
    es = 1
    for epoch in range(config.EPOCHS):
        if es > ESTOP:
            break

        df_train = shuffle(df_train)
        chunks = np.array_split(df_train, round(len(df_train)/SAVE))
    
        for chunk in chunks:
            train_dataset = dataset.BERTDataset(
                text=chunk.text.values,
                target=chunk.label.values
            )

            train_data_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.TRAIN_BATCH_SIZE,
                num_workers=4, shuffle=True
            )
            logger.info(f"Epoch = {epoch}")

            train_loss, train_acc = engine.train_fn(
                train_data_loader, model, optimizer, device, scheduler)

            for tag, parm in model.named_parameters():
                if parm.grad is not None:
                    writer.add_histogram(tag, parm.grad.data.cpu().numpy(), epoch)

            outputs, targets, val_loss, val_acc = engine.eval_fn(
                valid_data_loader, model, device)
            val_mcc = metrics.matthews_corrcoef(outputs, targets)
            logger.info(f"val_MCC_Score = {val_mcc:.4f}")

            outputs, targets, test_loss, test_acc = engine.eval_fn(
                test_data_loader, model, device)
            test_mcc = metrics.matthews_corrcoef(outputs, targets)
            logger.info(f"test_MCC_Score = {test_mcc:.4f}")

            logger.info(
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}")
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('loss/val', val_loss, epoch)
            writer.add_scalar('loss/test', test_loss, epoch)
            
            logger.info(
                f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")
            writer.add_scalar('acc/train', train_acc, epoch)
            writer.add_scalar('acc/val', val_acc, epoch)
            writer.add_scalar('acc/test', test_acc, epoch)
            
            logger.info(f"val_mcc={val_acc:.4f}, test_mcc={test_acc:.4f}")
            writer.add_scalar('mcc/val', val_mcc, epoch)
            writer.add_scalar('mcc/test', test_mcc, epoch)

            accuracy = metrics.accuracy_score(targets, outputs)
            logger.info(f"Accuracy Score = {accuracy:.4f}")
            
            if accuracy < 0.4:
                logger.info(f"Something is very wrong! Accuracy is only {accuracy:.4f} Stopping...")
                break

            if accuracy > best_accuracy:
                logger.info(f"Saving model with Accuracy Score = {accuracy:.4f}")
                if len(best_path) > 0 and os.path.exists(best_path):
                    #Delete previous best
                    os.remove(best_path)
                best_path = config.MODEL_PATH[:-4] + "." + str(round(accuracy*100, 2)) + ".bin"
                torch.save(model.state_dict(), best_path)
                best_accuracy = accuracy
                es = 0
            else:
                es += 1
                logger.info(f"Not improved for {es} times of {ESTOP}. Best so far - {best_accuracy:.4f}")

                if es > ESTOP:
                    logger.info(f"Early stopping with best accuracy: {best_accuracy:.4f} and accuracy for this epoch: {accuracy:.4f} ...")
                    break


if __name__ == "__main__":
    app.run(main)
