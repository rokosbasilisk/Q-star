import os
import random
import datasets
import nltk
from torch.utils.data import DataLoader

import transformers
from filelock import FileLock
from transformers import AutoConfig, AutoTokenizer, DataCollatorForSeq2Seq, T5Model
from transformers.utils import is_offline_mode
from prm_dataset import *

from parse_args import parse_args
from utils import (
    set_random_seed,
    prints,
    load_pretrained_model_config,
    load_pretrained_tokenizer,
    load_pretrained_model,
)
from reward_model import RewardModel, RewardModelTrainer


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main():
    args = parse_args()

    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    if args.source_prefix is None and args.model_name_or_path in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
        prints(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `", warning=True
        )

    set_random_seed(42)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    config = load_pretrained_model_config(args, AutoConfig)
    tokenizer = load_pretrained_tokenizer(args, AutoTokenizer)
    reward_transformer = load_pretrained_model(args, T5Model, config, tokenizer, reward_model=True)

    ##########################################
    # add datasets here

    train_dataset = None
    eval_dataset = None

    prints(f'len(train_dataset) = {len(train_dataset)}')
    prints(f'len(eval_dataset) = {len(eval_dataset)}')

    ##########################################

    for index in random.sample(range(len(train_dataset)), 1):
        prints(f"Sample {index} of the training set: {train_dataset[index]}")


    # data loaders for training the reward function

    train_path = "prmdata/train.jsonl"
    eval_path = "prmdata/test.jsonl"


    train_dataset = RewardModelDataset(train_path)
    validation_dataset = RewardModelDataset(eval_path)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    eval_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False)


    # initialize the trainers
    model = RewardModel(args=args, reward_transformer=reward_transformer)
    model_trainer = RewardModelTrainer(args=args, 
    model=model,
    tokenizer=tokenizer, 
    rew_train_dataloader=train_dataloader, 
    rew_eval_dataloader=eval_dataloader )
    model_trainer.train()


if __name__ == "__main__":
    main()
