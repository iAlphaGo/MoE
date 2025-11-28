import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from moe import MoE  # 假设 MoE 类已经定义好
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import ADASYN  # 使用 ADASYN 处理类别不平衡
import matplotlib.pyplot as plt
import seaborn as sns


# 加载 CSV 文件
def load_csv(file_path):
    df = pd.read_csv(file_path)
    df.fillna(df.mean(), inplace=True)  # 填充缺失值
    categorical_columns = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=categorical_columns)
    label_encoder = LabelEncoder()
    df['class'] = label_encoder.fit_transform(df['class'])
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
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
features, labels = load_csv('KDDTest+.csv')

# 输出类别分布
print("\nClass Distribution:")
unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} samples")

# 输出正常流量和异常流量的数量
normal_count = counts[0] if len(counts) > 0 else 0
anomaly_count = counts[1] if len(counts) > 1 else 0
print(f"Normal traffic: {normal_count} samples")
print(f"Anomaly traffic: {anomaly_count} samples")

# 处理类别不平衡
adasyn = ADASYN(random_state=42)
features_resampled, labels_resampled = adasyn.fit_resample(features.numpy(), labels.numpy())
features_resampled = torch.tensor(features_resampled, dtype=torch.float32)
labels_resampled = torch.tensor(labels_resampled, dtype=torch.long)

# 划分数据集为训练集和测试集，确保类别平衡
X_train, X_test, y_train, y_test = train_test_split(features_resampled, labels_resampled, test_size=0.2,
                                                    random_state=42, stratify=labels_resampled)

# 创建训练集和测试集的数据集
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)

# 创建训练集和测试集的数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化 MoE 模型
model = MoE(input_size=features.shape[1], output_size=2, num_experts=10, hidden_size=64, k=4, noisy_gating=True)
model = model.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 训练模式
model.train()
train_losses = []
for epoch in range(10):
    epoch_loss = 0.0
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        y_hat, aux_loss = model(inputs)
        loss = criterion(y_hat, labels) + aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    epoch_loss /= len(train_dataloader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

# 绘制损失曲线
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

# 评估模式
model.eval()
all_labels, all_preds = [], []
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        y_hat, _ = model(inputs)
        preds = torch.argmax(y_hat, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# 计算评估指标
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = conf_matrix.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

# 输出评估指标
print("\nEvaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"FPR:       {fpr:.4f}")

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# 保存模型参数
torch.save(model.state_dict(), 'moe_model.pth')