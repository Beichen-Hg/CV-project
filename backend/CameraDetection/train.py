# 训练和验证模块
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR


def train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss}')

        model.eval()  # 设置模型为评估模式
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():  # 在验证阶段不计算梯度
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据和标签移动到 GPU
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()

        val_loss = running_loss / len(val_loader)
        print(f'Accuracy on validation set: {100 * correct / total}%')
        print(f'val_Loss: {val_loss / len(val_loader)}')
        print('-' * 10)

        # 动态调整学习率
        scheduler.step(val_loss)

        if correct / total > best_accuracy:
            best_accuracy = correct / total
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, 'test_best_model_weights.pth')

    print(f'Best val Accuracy: {best_accuracy}%')