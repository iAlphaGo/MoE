import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from moe import MoE  # 假设 MoE 类已经定义好
import torch.optim as optim
import torch.nn as nn


# 加载 CSV 文件
def load_csv(file_path):
    # 加载 CSV 文件
    df = pd.read_csv(file_path)

    # 查看数据的前几行
    print("Data head:")
    print(df.head())

    # 查看每列的数据类型
    print("Data types:")
    print(df.dtypes)

    # 检查缺失值
    print("Missing values:")
    print(df.isnull().sum())

    # 处理缺失值：删除包含缺失值的行
    df = df.dropna()

    # 假设最后一列是标签，其余列是特征
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values

    # 转换为 PyTorch 张量
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # 保存清洗后的数据
    df.to_csv('cleaned_dataset.csv', index=False)

    return features, labels


# 定义一个自定义数据集类
class MyDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            features = self.transform(features)

        return features, label


# 加载和预处理 CSV 文件
features, labels = load_csv('entry01.weka.allclass.csv')

# 创建数据集
dataset = MyDataset(features, labels)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 实例化 MoE 层
model = MoE(input_size=262, output_size=2, num_experts=10, hidden_size=64, k=4, noisy_gating=True)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模式
model.train()
for epoch in range(10):  # 假设训练 100 个 epoch
    for inputs, labels in dataloader:
        # 前向传播
        y_hat, aux_loss = model(inputs)

        # 计算总损失
        loss = criterion(y_hat, labels) + aux_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# # 评估模式
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for inputs, labels in dataloader:
#         y_hat, _ = model(inputs)
#         _, predicted = torch.max(y_hat, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     accuracy = 100 * correct / total
#     print(f'Accuracy: {accuracy}%')
# 评估模式
model.eval()
all_labels = []
all_predicted = []
with torch.no_grad():
    for inputs, labels in dataloader:
        y_hat, _ = model(inputs)
        _, predicted = torch.max(y_hat, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())

# 计算评估指标
accuracy = accuracy_score(all_labels, all_predicted)
precision = precision_score(all_labels, all_predicted, average='weighted')
recall = recall_score(all_labels, all_predicted, average='weighted')
f1 = f1_score(all_labels, all_predicted, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_predicted)
tn, fp, fn, tp = conf_matrix.ravel()
false_positive_rate = fp / (fp + tn)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'False Positive Rate: {false_positive_rate:.4f}')
# 保存模型参数
torch.save(model.state_dict(), 'moe_model.pth')

# 加载模型参数
model.load_state_dict(torch.load('moe_model.pth'))