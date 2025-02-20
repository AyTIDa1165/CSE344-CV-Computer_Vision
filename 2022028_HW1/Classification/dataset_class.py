import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, class_names, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.images = []
        self.labels = []
        self.transform = transform
        self.target_transform = target_transform

        for i in range(len(class_names)):
            class_dir = os.path.join(img_dir, class_names[i])
            try:
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_names[i], img_name)
                    self.images.append(img_path)
                    self.labels.append(i)

            except FileNotFoundError:
                print(f"Directory for label '{class_names[i]}' not found")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.images[idx])
        image = read_image(image_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def get_dataloaders(dataset, batch_size):
    train_indices, val_indices = train_test_split(
        list(range(dataset.__len__())), test_size=0.2, stratify=dataset.labels)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
    return train_dataloader, val_dataloader

# class_names = ['amur_leopard', 'amur_tiger', 'birds', 'black_bear', 'brown_bear', 'dog', 'roe_deer', 'sika_deer', 'wild_boar', 'people' ]

# dataset = CustomImageDataset('russian-wildlife-dataset/Cropped_final', class_names)
# print(dataset.__len__())