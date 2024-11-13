#模型定义模块
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes):
    resnet50 = models.resnet50(pretrained=True)
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, 10)
    return resnet50