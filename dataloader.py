import json
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

class CustomDataset(Dataset):
    def __init__(self, file_path,args,tokenizer):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 确保行不为空
                    self.data.append(json.loads(line))  # 解析 JSON 并添加到列表中
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        labels = self.data[idx]['label']

        if len(text) > 510:
            text = text[:510 ]
            labels = labels[:510]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + text + ["[SEP]"])
        attention_mask = [1] * len(tmp_input_ids)
        input_ids = tmp_input_ids + [0] * (512 - len(tmp_input_ids))#padding操作
        attention_mask = attention_mask + [0] * (512 - len(tmp_input_ids))
        labels = [self.args.label2id[label] for label in labels]
        labels = [0] + labels + [0] + [0] * (512 - len(tmp_input_ids))

        input_ids = torch.tensor(np.array(input_ids))
        attention_mask = torch.tensor(np.array(attention_mask))
        labels = torch.tensor(np.array(labels))

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return data




