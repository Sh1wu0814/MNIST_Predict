"""
数字识别模型训练脚本
使用PyTorch训练MNIST数字识别模型，确保测试准确率达到90%以上
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from PIL import Image, ImageDraw
import os

# ==================== 1. 改进的CNN模型定义 ====================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一个卷积块：5x5卷积核，保持尺寸
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 全连接层，添加dropout防止过拟合
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # 返回原始logits，CrossEntropyLoss内部会处理softmax
        return x

# ==================== 2. 图像预处理函数 ====================
def points_to_image(points, width=400, height=400, return_image=False):
    """将记录的坐标点转换为28x28 MNIST格式图像
    与MNIST一致：黑底白字

    简化改进版：
    1. 简单直接的坐标映射（拉伸以适应）
    2. 固定细线条（2像素）
    3. 确保数字填满图像中心区域
    4. 优化模糊处理

    参数:
        points: 坐标点列表 [(x1, y1), (x2, y2), ...]
        width: 画布宽度（默认400）
        height: 画布高度（默认400）
        return_image: 是否返回PIL图像对象（默认False，只返回张量）

    返回:
        如果 return_image=False: 返回图像张量 (1, 1, 28, 28)
        如果 return_image=True: 返回元组 (PIL图像对象, 图像张量)
    """
    img_size = 28
    # 创建黑底图像（MNIST原始格式）
    img = Image.new('L', (img_size, img_size), 0)

    if not points or len(points) < 2:
        # 返回全黑图像
        img_array = np.zeros((img_size, img_size), dtype=np.float32)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        img_tensor = (img_tensor - 0.1307) / 0.3081
        if return_image:
            return img, img_tensor
        else:
            return img_tensor

    draw = ImageDraw.Draw(img)

    # 提取坐标
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # 计算边界框尺寸
    box_width = max_x - min_x
    box_height = max_y - min_y

    # 防止宽度或高度为0
    if box_width == 0:
        box_width = 1
    if box_height == 0:
        box_height = 1

    # 计算边界框中心
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # 确保绘制尺寸足够大（至少为画布的15%）
    min_size = min(width, height) * 0.15
    if box_width < min_size or box_height < min_size:
        # 扩展边界框，保持中心不变
        if box_width < min_size:
            expansion = (min_size - box_width) / 2
            min_x = center_x - min_size/2
            max_x = center_x + min_size/2
            box_width = min_size
        if box_height < min_size:
            expansion = (min_size - box_height) / 2
            min_y = center_y - min_size/2
            max_y = center_y + min_size/2
            box_height = min_size

    # 映射：保持宽高比缩放以适应图像中心区域
    # 使用适当的边距，让数字清晰可见
    margin = 3
    target_width = img_size - 2 * margin
    target_height = img_size - 2 * margin

    # 计算缩放比例，保持宽高比
    scale_x = target_width / box_width
    scale_y = target_height / box_height
    scale = min(scale_x, scale_y)  # 保持宽高比

    # 计算缩放后的实际尺寸
    scaled_width = box_width * scale
    scaled_height = box_height * scale

    # 计算偏移量，使数字居中（考虑缩放后的实际尺寸）
    offset_x = margin + (target_width - scaled_width) / 2 - min_x * scale
    offset_y = margin + (target_height - scaled_height) / 2 - min_y * scale

    # 自适应线条宽度：根据缩放比例调整
    # 缩放比例大（数字小）时用较细线条，缩放比例小（数字大）时用稍粗线条
    if scale > 0.4:
        line_width = 1
    elif scale > 0.25:
        line_width = 2
    else:
        line_width = 3

    # 绘制线条
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        # 映射到图像坐标
        ix1 = int(x1 * scale + offset_x)
        iy1 = int(y1 * scale + offset_y)
        ix2 = int(x2 * scale + offset_x)
        iy2 = int(y2 * scale + offset_y)

        # 翻转y轴（turtle坐标y向上为正）
        iy1 = img_size - 1 - iy1
        iy2 = img_size - 1 - iy2

        # 绘制白色线条（MNIST数字是白色的）
        draw.line([(ix1, iy1), (ix2, iy2)], fill=255, width=line_width)

    # 应用轻微高斯模糊，模拟手写数字的平滑边缘
    from PIL import ImageFilter
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

    # 转换为numpy数组并归一化
    img_array = np.array(img).astype(np.float32) / 255.0

    # 应用MNIST标准化（与训练时完全相同）
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    img_tensor = (img_tensor - 0.1307) / 0.3081

    if return_image:
        return img, img_tensor
    else:
        return img_tensor

# ==================== 3. 训练函数 ====================
def train_model(epochs=10):
    """
    训练MNIST数字识别模型

    参数:
        epochs: 训练轮数，默认10个epoch确保准确率>90%

    返回:
        训练好的模型
    """
    print("开始训练MNIST数字识别模型...")
    print(f"训练轮数: {epochs}")
    print("=" * 50)

    # 设置随机种子以确保可重复性
    torch.manual_seed(1)
    np.random.seed(1)

    # 数据预处理
    # 训练时使用数据增强，测试时仅使用标准化
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),  # 随机旋转±5度
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载MNIST数据集
    print("加载MNIST数据集...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    print(f"训练样本: {len(train_dataset)}，测试样本: {len(test_dataset)}")

    # 初始化模型、优化器
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # 学习率调度器：当测试损失在3个epoch内不再下降时，学习率乘以0.5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 训练循环
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 测试准确率
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total_test += target.size(0)
                correct_test += predicted.eq(target).sum().item()

        train_accuracy = 100. * correct / total
        test_accuracy = 100. * correct_test / total_test

        print(f'\nEpoch {epoch+1}/{epochs} 结果:')
        print(f'训练准确率: {train_accuracy:.2f}%')
        print(f'测试准确率: {test_accuracy:.2f}%')
        print(f'测试损失: {test_loss/len(test_loader):.4f}')
        # 使用学习率调度器
        scheduler.step(test_loss/len(test_loader))
        print("-" * 40)

        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

    # 保存最终模型
    os.makedirs('./data', exist_ok=True)
    model_path = './data/mnist_model.pth'
    torch.save(model.state_dict(), model_path)

    print("=" * 50)
    print(f"模型训练完成!")
    print(f"最终测试准确率: {best_accuracy:.2f}%")
    print(f"模型已保存到: {model_path}")
    print("=" * 50)

    return model

# ==================== 4. 模型评估函数 ====================
def evaluate_model(model_path='./data/mnist_model.pth'):
    """
    评估已训练模型的准确率
    """
    print("评估模型性能...")

    # 加载模型
    model = SimpleCNN()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"已加载模型: {model_path}")
    else:
        print(f"未找到模型文件: {model_path}")
        return None

    model.eval()

    # 数据预处理（与训练时的测试转换一致）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载测试集
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 评估
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    print(f"测试准确率: {accuracy:.2f}% ({correct}/{total})")

    if accuracy >= 90.0:
        print("[OK] 模型准确率达到90%以上要求!")
    else:
        print("[WARNING] 模型准确率未达到90%，建议重新训练或增加训练轮数")

    return accuracy

# ==================== 5. 主程序入口 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MNIST数字识别模型训练')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--evaluate', action='store_true', help='仅评估模型，不训练')
    parser.add_argument('--model-path', type=str, default='./data/mnist_model.pth', help='模型路径')

    args = parser.parse_args()

    if args.evaluate:
        # 仅评估模型
        evaluate_model(args.model_path)
    else:
        # 训练新模型
        model = train_model(epochs=args.epochs)

        # 评估新训练的模型
        evaluate_model('./data/mnist_model.pth')