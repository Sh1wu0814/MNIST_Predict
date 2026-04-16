# 数字识别模型训练程序

一个使用PyTorch训练MNIST数字识别模型的简单程序。

## 功能特点
- 使用PyTorch训练MNIST数字识别模型（简单的CNN架构）
- 包含预训练模型 `mnist_model.pth`，准确率达到90%以上
- 简单的训练脚本，支持自定义训练轮数
- 自动下载MNIST数据集进行训练

## 包含文件
本项目仅包含两个核心文件：
1. **train_model.py** - 模型训练脚本
2. **data/mnist_model.pth** - 预训练模型文件

## 系统要求
- Python 3.8+
- PyTorch 1.9.0+
- torchvision 0.10.0+

## 项目结构
数字识别/
├── train_model.py           # 模型训练脚本
├── data/                    # 数据目录
│   └── mnist_model.pth     # 预训练模型
└── README.md               # 说明文档



## 快速开始
- 安装依赖：`pip install torch torchvision`
- 使用预训练模型：`mnist_model.pth` 可直接用于数字识别
- 训练新模型：`python train_model.py --epochs 10`

## 模型架构
简单的CNN架构（28×28输入 → 卷积层 → 池化层 → 全连接层 → 10个输出）

## 训练参数
- MNIST数据集（60,000训练样本 + 10,000测试样本）
- Adam优化器，交叉熵损失
- 默认10个epoch，预期准确率>90%
