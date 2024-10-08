import os

class Config:
    MODEL_PATH = "D:\\Models\\bert-base-chinese\\"

class Modelconfig:
    def __init__(self):
        self.output_dir = "checkpoint\\"
        self.max_seq_len = 510
        self.epochs = 3
        self.train_batch_size = 12
        self.dev_batch_size = 12
        self.bert_learning_rate = 3e-5
        self.crf_learning_rate = 3e-3
        self.adam_epsilon = 1e-8
        self.weight_decay = 0.01
        self.warmup_proportion = 0.01
        self.save_step = 500
        self.overlap = 50
        self.bert_dir = "D:\\Models\\bert-base-chinese\\"
        self.num_labels = 14
        label_file = 'Data\\label.txt'

        id2label = {}

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 按照冒号分割每一行，去除前后空格
                label, label_id = line.strip().split(':')
                id2label[int(label_id)] = label.strip()

        self.id2label = id2label
        label_file = 'Data\\label.txt'

        label2id = {}

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 按照冒号分割每一行，去除前后空格
                label, label_id = line.strip().split(':')
                label2id[label] = int(label_id)
        self.label2id = label2id

