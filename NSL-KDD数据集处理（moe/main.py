import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.optim as optim
import torch.nn as nn
from moe import MoE  # 假设 MoE 类已经定义好


# 加载 CSV 文件
def load_csv(file_path):
    # 加载 CSV 文件
    df = pd.read_csv(file_path)

    # 去除列名中的前导空格
    df.columns = df.columns.str.strip()

    # 检查缺失值
    print("Missing values:")
    print(df.isnull().sum())

    # 处理缺失值：删除包含缺失值的行
    df = df.dropna()

    # 将标签转换为二分类（正常流量为 0，异常流量为 1）
    label_column = 'class'
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found. Available columns: {df.columns}")

    # 定义正常流量标签（根据 NSL-KDD 数据集的实际情况）
    normal_label = 'normal'
    df[label_column] = df[label_column].apply(
        lambda x: 0 if str(x).strip().lower() == normal_label.lower() else 1
    )

    # 处理分类特征（示例列名，根据实际数据调整）
    categorical_columns = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=categorical_columns)

    # 提取特征和标签
    features = df.drop(label_column, axis=1).values
    labels = df[label_column].values

    # 处理非数值型和无穷大值
    features = np.where(np.isinf(features), np.nan, features)
    features = np.where(np.abs(features) > 1e10, np.nan, features)
    features = np.nan_to_num(features, nan=np.nanmean(features, axis=0))

    # 数据标准化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 转换为 PyTorch 张量
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # 保存清洗后的数据
    df.to_csv('cleaned_nslkdd.csv', index=False)

    return features, labels


# 自定义数据集类（与第一段代码保持一致）
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


# 加载和预处理数据
features, labels = load_csv('KDDTest+.csv')

# 输出正常和异常流量数量
normal_count = (labels == 0).sum().item()
anomaly_count = (labels == 1).sum().item()
print("\nClass Distribution:")
print(f"Normal traffic: {normal_count} samples")
print(f"Anomaly traffic: {anomaly_count} samples")

# 划分数据集（保持类别平衡）
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# 创建数据加载器
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化模型（二分类输出）
model = MoE(
    input_size=features.shape[1],
    output_size=2,  # 二分类
    num_experts=10,
    hidden_size=64,
    k=4,
    noisy_gating=True
).to(device)

# 训练配置（保持与第一段代码一致）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
model.train()
for epoch in range(200):
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        y_hat, aux_loss = model(inputs)
        loss = criterion(y_hat, labels) + aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# 评估模型
model.eval()
all_labels, all_preds = [], []
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        y_hat, _ = model(inputs)
        preds = torch.argmax(y_hat, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# 计算指标（使用binary模式）
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary', pos_label=1)
recall = recall_score(all_labels, all_preds, average='binary', pos_label=1)
f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)
conf_matrix = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = conf_matrix.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

# 输出结果
print("\nEvaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"FPR:       {fpr:.4f}")

# 保存模型
torch.save(model.state_dict(), 'moe_model_nslkdd.pth')