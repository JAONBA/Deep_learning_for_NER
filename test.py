from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import BertModel, BertNer
from config import Config, Modelconfig
from dataloader import CustomDataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import torch
from tqdm import tqdm

if __name__ == '__main__':
    # 加载测试数据
    args = Modelconfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    test_dataset = CustomDataset("Data\\test.txt", args, tokenizer)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=2)

    # 加载模型
    model = BertNer(args).to(device)
    model.load_state_dict(torch.load('checkpoint\\pytorch_model_ner.bin'))
    model.eval()

    preds = []
    trues = []

    for step, batch_data in enumerate(tqdm(test_loader)):
        for key, value in batch_data.items():
            batch_data[key] = value.to(device)

        input_ids = batch_data["input_ids"]
        attention_mask = batch_data["attention_mask"]
        labels = batch_data["labels"]
        output = model(input_ids, attention_mask, labels)
        logits = output.logits
        attention_mask = attention_mask.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        batch_size = input_ids.size(0)
        for i in range(batch_size):
            length = sum(attention_mask[i])  # 计算有效长度
            logit = logits[i][1:length]  # 去掉[CLS]标记
            logit = [args.id2label[i] for i in logit]
            label = labels[i][1:length]  # 去掉[CLS]标记
            label = [args.id2label[i] for i in label]

            # 将每个字符及其对应的标签添加到输出中
            for j in range(length - 1):  # length-1是因为我们已经去掉了[CLS]
                char = tokenizer.decode(input_ids[i][j + 1].item())  # 通过input_ids获取字符
                preds.append(f"{char}:{logit[j]}")
                trues.append(f"{char}:{label[j]}")

    # 将 preds 和 trues 转换为二进制格式并计算分类报告
    mlb = MultiLabelBinarizer()
    trues_binary = mlb.fit_transform([trues])
    preds_binary = mlb.transform([preds])

    report = classification_report(trues_binary, preds_binary)
    print(report)

    # 输出每个字符及其对应的标签
    # for pred, true in zip(preds, trues):
    #     print(f"Predicted: {pred}, True: {true}")
