# 主程序模块
import torch
from model import get_model
from data_loader import get_data_loaders
from train import train_model
from torch import nn, optim


def main():
    batch_size = 64
    num_epochs = 40
    num_classes = 10

    # 获取 dataloader
    train_loader, val_loader = get_data_loaders('MY_data', batch_size)

    # 获取模型
    model = get_model(num_classes)

    # 定义优化器
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001, betas=(0.9, 0.999))

    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 定义学习率调度器
    # 修改学习率调度器的定义
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        threshold=0.01,
        patience=5
    )

    # 传入参数，开始训练模型并保存好训练好的参数
    train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, scheduler)

if __name__ == '__main__':
    main()