# -*- coding: utf-8 -*-

import torch
import json
from torch.nn.utils.rnn import pad_sequence
from model_train import BiLSTM_CRF
# 假设已定义 BiLSTM_CRF 类和其他必要的函数

# 加载模型
# 加载模型
def load_model(model_path, vocab_size, embedding_dim, hidden_dim, num_classes):
    model = BiLSTM_CRF(vocab_size, embedding_dim, hidden_dim, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    return model

# 准备输入数据
def prepare_input(text, vocab):
    # 将每个字符进行编码
    encoded_text = torch.tensor([vocab.get(char, 0) for char in text])  # 使用词汇表编码字符
    return encoded_text.unsqueeze(0)  # 增加批量维度

# 进行预测
def predict(model, text, vocab, device):
    model.eval()
    with torch.no_grad():
        input_tensor = prepare_input(text, vocab).to(device)  # 对输入文本逐字编码
        # 获取模型输出
        emissions = model(input_tensor)

        # 创建掩码
        mask = (input_tensor != 0)

        # 使用 viterbi_decode 方法获取预测标签
        predicted_labels = model.crf.viterbi_decode(emissions, mask=mask)
    return predicted_labels[0]  # 返回解码后的标签

# 主函数
def main():
    # 加载词汇表
    with open('vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model('bilstm_crf_model.pth', len(vocab) + 1, 128, 256, 14).to(device)  # 移动模型到设备

    # 输入文本
    text = "【药品名称】通用名称：六味地黄丸(浓缩丸)商品名称：同仁堂英文名称：LiuWeiDiHuangWan(TongRenTang)汉语拼音：Liuwei Dihuang Wan【成份】熟地黄、山茱萸(制)、山药、牡丹皮、茯苓、泽泻。"

    # 进行预测
    predicted_labels = predict(model, text, vocab, device)

    # 标签映射字典
    label_mapping = {
        0: 'O',  # 非实体
        1: '药品',
        2: '药物成分',
        3: '药物剂型',
        4: '药物性味',
        5: '中药功效',
        6: '症状',
        7: '人群',
        8: '食物分组',
        9: '食物',
        10: '疾病',
        11: '证候',
        12: '疾病分组',
        13: '药品分组'
    }

    # 输出结果：char:label 形式
    for char, label_idx in zip(text, predicted_labels):
        label = label_mapping.get(label_idx, 'O')  # 获取实体标签
        print(f'{char}: {label}')

if __name__ == "__main__":
    main()