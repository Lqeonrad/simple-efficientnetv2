
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import config
from utils import train_one_epoch, validate
from tqdm import tqdm  # 导入进度条库

# 配置设备
device = torch.device("cpu")  # 因为没有 CUDA，所以使用 CPU

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将 CIFAR-10 的 32x32 图片调整为 224x224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# 使用 EfficientNetV2 预训练模型
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
num_classes = config.NUM_CLASSES  # CIFAR-10 的类别数
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

# 开始训练
for epoch in range(config.EPOCHS):
    print(f"\nEpoch [{epoch + 1}/{config.EPOCHS}]")

    # 训练阶段
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

    # 验证阶段
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # 打印训练和验证的损失和准确率
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# 保存模型
torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

