import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from TorchCRF import CRF
from sklearn.model_selection import train_test_split
from collections import defaultdict

# 数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

    @staticmethod
    def collate_fn(batch):
        texts, labels = zip(*batch)
        texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)  # 使用0填充
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)  # 使用-1填充
        return texts_padded, labels_padded

# BiLSTM + CRF模型
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.crf = CRF(num_classes)

    def forward(self, x, tags=None):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.fc(lstm_out)

        # 创建掩码
        mask = (x != 0)  # 假设填充的值是0

        if tags is not None:
            mask = (x != 0)
            loss = -self.crf(emissions, tags, mask=mask)
            return loss.mean()
        else:
            return emissions  # 只返回 emissions 用于预测

# 读取数据并构建词汇表
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    return texts, labels

def build_vocab(texts):
    word_count = defaultdict(int)
    for text in texts:
        for word in text:
            word_count[word] += 1
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_count.items())}  # 0用作填充
    return vocab

def encode_texts(texts, vocab):
    return [[vocab.get(word, 0) for word in text] for text in texts]


def save_vocab(vocab, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)
# 主函数
label_mapping = {
    '0': 0,
    '药品': 1,
    '药物成分': 2,
    '药物剂型': 3,
    '药物性味': 4,
    '中药功效': 5,
    '症状': 6,
    '人群': 7,
    '食物分组': 8,
    '食物': 9,
    '疾病': 10,
    '证候': 11,
    '疾病分组': 12,
    '药品分组': 13
}

# 标签编码函数
def encode_labels(labels, label_mapping):
    return [[label_mapping.get(label, 0) for label in lbl] for lbl in labels]  # 默认为0

# 主函数
def main():
    # 数据准备
    texts, labels = load_data('Data\\output_list.json')  # 替换为你的数据文件路径
    vocab = build_vocab(texts)
    encoded_texts = encode_texts(texts, vocab)

    # 编码标签
    encoded_labels = encode_labels(labels, label_mapping)  # 将标签转换为数值

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(encoded_texts, encoded_labels, test_size=0.2, random_state=42)

    # 数据加载
    train_dataset = TextDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=TextDataset.collate_fn)

    # 模型参数
    vocab_size = len(vocab) + 1  # 加1以便于填充
    embedding_dim = 128
    hidden_dim = 256
    num_classes = len(label_mapping)  # 标签的唯一值数

    # 检查CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型初始化
    model = BiLSTM_CRF(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model.train()
    for epoch in range(15):  # 迭代10轮
        total_loss = 0
        for batch_idx, (texts, labels) in enumerate(train_loader):
            texts, labels = texts.to(device), labels.to(device)  # 将数据移动到GPU
            optimizer.zero_grad()
            loss = model(texts, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:  # 每10个batch打印一次
                print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}')

        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1} finished. Average Loss: {average_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'bilstm_crf_model.pth')

if __name__ == "__main__":
    main()

