#数据加载模块
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FruitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for i in range(10):  # 假设有10个类别
            folder_path = os.path.join(self.root_dir, str(i))
            image_folder = os.listdir(folder_path)
            for image_name in image_folder:
                self.image_paths.append((os.path.join(folder_path, image_name), i))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loaders(root_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = FruitDataset(os.path.join(root_dir, 'train'), transform=transform)
    val_dataset = FruitDataset(os.path.join(root_dir, 'valid'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader