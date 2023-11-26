#!/usr/bin/env python
# coding: utf-8

from bisect import bisect_left
from itertools import accumulate
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gc
import json
import random
import torch
import wandb

class RewardModelDataset(Dataset):
    
    def __init__(self, json_file):
        self.data = [sample for sample in self.load_data_from_file(json_file) if len(sample["label"]["steps"]) > 0]
        
        self.total_length = sum(len(sample["label"]["steps"]) for sample in self.data)
        
        self.slots = list(accumulate([len(sample["label"]["steps"]) for sample in self.data]))
        self.ctx_target_pairs = []
        
        def find_pos(numbers, x):
            if x < numbers[0]:
                return 0  # Insert at the beginning
            elif x > numbers[-1]:
                return len(numbers)  # Insert at the end
            else:
                return bisect_left(numbers, x)
        
        for idx in tqdm(range(self.total_length)):
            slot_idx = find_pos(self.slots, idx)
            sample_idx = self.slots[slot_idx] - idx 
            sample = self.data[sample_idx]

            question = sample["question"]["problem"]
            steps = sample["label"]["steps"][:sample_idx + 1]
            context = question+"[ANS]"

            targets = []

            for step in steps[:-1]:
                completion = random.choice(step["completions"])
                context += f"  [SEP]{completion['text']} <[RATING]> {completion['rating']}"

            final_ctx = random.choice(steps[-1]["completions"])

            context += f"  [SEP]{final_ctx['text']} <[RATING]>"
            target = ({-1: 0.0, 0: 0.5, 1: 1.0 }.get(final_ctx['rating'], 0.0)*idx)/self.total_length # trick to convert feedback to reward
            self.ctx_target_pairs.append({"context": context, "target": target})


    def load_data_from_file(self, json_file):
        with open(json_file, 'r') as file:
            data = [json.loads(line) for line in file]
        return data

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        return self.ctx_target_pairs[idx]

    def __getitem__(self, idx):
        return self.ctx_target_pairs[idx]
