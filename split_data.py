import json
import random

import json
import random

# 读取 JSON 文件
with open('Data\\output_list.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


# 打乱数据顺序
random.shuffle(data)
# 划分比例
train_size = int(0.7 * len(data))
dev_size = int(0.15 * len(data))
test_size = len(data) - train_size - dev_size

# 划分数据集
train_data = data[:train_size]
dev_data = data[train_size:train_size + dev_size]
test_data = data[train_size + dev_size:]

# 写入 train.txt
with open('Data\\train.txt', 'w', encoding='utf-8') as f:
    for i in train_data:
        json.dump(i, f, ensure_ascii=False)
        f.write('\n')  # 添加换行符

# 写入 dev.txt
with open('Data\\dev.txt', 'w', encoding='utf-8') as f:
    for j in dev_data:
        json.dump(j, f, ensure_ascii=False)
        f.write('\n')  # 添加换行符

# 写入 test.txt
with open('Data\\test.txt', 'w', encoding='utf-8') as f:
    for k in test_data:
        json.dump(k, f, ensure_ascii=False)
        f.write('\n')  # 添加换行符

print("数据集划分完成！")
