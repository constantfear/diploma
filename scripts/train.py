import random
import torch
import numpy as np
from transformers import MBart50Tokenizer
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
# from modules.utils import set_seed
from modules.new_model import My_MBart
import argparse
import logging
import wandb
import traceback


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

TEXT_COLUMN = "input_text"
SUMMARY_COLUMN = "title"
MAX_INPUT_LENGTH = 1024
MAX_METADATA_LENGTH = 128
MAX_TARGET_LENGTH = 128

cluster_centers = pd.read_csv('./data/cluster_centers.csv')
cluster_centers = cluster_centers.drop('cluster_id', axis=1)
cluster_centers = cluster_centers.values


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mbart_preprocess(examples, tokenizer):
    # Токенизация основного текста
    inputs = examples[TEXT_COLUMN]
    targets = examples[SUMMARY_COLUMN]
    cluster = examples['cluster']

    text_enc = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # Токенизация заголовка (labels)
    label_enc = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )

    return {
        "input_ids": text_enc["input_ids"],
        "attention_mask": text_enc["attention_mask"],
        "meta_embs": cluster_centers[cluster],
        "labels": label_enc["input_ids"]
    }


def preprocess_function(examples, tokenizer):
    inputs = examples[TEXT_COLUMN]
    targets = examples[SUMMARY_COLUMN]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(args: argparse.Namespace):
    wandb.init(project="headline_generation", name=f"{args.logger_name}_{args.seed}")
    file_handler = logging.FileHandler(f"logs/{args.logger_name}_{args.seed}.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f'Random seed is: {args.seed}')
    set_seed(args.seed)
    logger.info(f'Pretrained Model Name: {args.model_name}')

    tokenizer = MBart50Tokenizer.from_pretrained(args.model_name)

    logger.info('Read data...')
    df_texts = pd.read_csv('./data/new_train_dataset.csv')
    # df_texts = df_texts.iloc[:1000]
    logger.info(f'Train Type: {args.train_type}')
    if args.train_type == 'add_cluster':
        injection_type = args.injection_type
        model = My_MBart.from_pretrained_combined(args.model_name, use_meta=True, injection_type=injection_type)
        logger.info('Process data...')

        df_texts['input_text'] = df_texts['text']

        dataset = Dataset.from_pandas(df_texts)
        tokenized_datasets = dataset.map(mbart_preprocess,
                                         fn_kwargs={'tokenizer': tokenizer},
                                         remove_columns=df_texts.columns.tolist())
        # tokenized_datasets.set_format(type='torch', columns=['input_ids',
        #                                                      'attention_mask',
        #                                                      'meta_input_ids',,
        #                                                      'labels'])

        new_data = tokenized_datasets.train_test_split(test_size=0.1)
    elif args.train_type == 'default':
        model = My_MBart.from_pretrained_combined(args.model_name, use_meta=False)
        logger.info('Process data...')
        df_texts['input_text'] = df_texts['text']
        dataset = Dataset.from_pandas(df_texts)
        tokenized_datasets = dataset.map(preprocess_function,  fn_kwargs={'tokenizer': tokenizer})
        new_data = tokenized_datasets.train_test_split(test_size=0.1)
    else:
        logger.error("UNKNOWN TRAINING TYPE! \n Available options:\n - default\n - add_cluster")
        return
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    logger.info('Start training model...')

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=100,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=new_data['train'],
        eval_dataset=new_data['test'],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    logger.info('Finish training model...')

    model.save_pretrained(f'./models/{args.output_dir}')
    logger.info(f'Model saved in models/{args.output_dir}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="facebook/mbart-large-50")
    parser.add_argument("--logger_name", default="train_logs_mbart-large-50")
    parser.add_argument("--train_type", default="default")
    parser.add_argument("--injection_type", default="0")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", "-bs", default=16, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--output_dir", default="bart_headline_model")

    args = parser.parse_args()

    try:
        main(args)
    except Exception:
        logger.error(traceback.print_exc())
