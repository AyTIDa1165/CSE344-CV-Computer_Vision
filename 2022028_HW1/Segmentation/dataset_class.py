import os
import torch
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

class CamVidDataset(Dataset):
    def __init__(self, img_dir, label_dir, class_dict_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.images = []
        self.labels = []
        self.transform = transform
        self.target_transform = target_transform

        df = pd.read_csv(class_dict_dir)
        self.class_dict = {(item['r'], item['g'], item['b']): index for index, item in df.iterrows()}
        try:
            for img in os.listdir(img_dir):
                self.images.append(img)
            for label in os.listdir(label_dir):
                self.labels.append(label)

            self.images.sort()
            self.labels.sort()

            if len(self.images) != len(self.labels):
                raise ValueError("Number of images does not match the number of labels.")

        except FileNotFoundError:
            print(f"Directory for label not found")

        except ValueError as e:
            print(f'Error: {e}')

    def label_encode(self, label):
        (_, height, width) = label.shape
        encoded_label = torch.zeros((height, width), dtype=torch.long)

        for (rgb, label_num) in self.class_dict.items():
            mask = torch.all(label == torch.tensor(rgb).view(-1, 1, 1), dim=0)
            encoded_label[mask] = label_num
        
        return encoded_label


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        image = read_image(img_path)
        label = read_image(label_path)

        if self.transform:
            image = self.transform(image)
            label = self.target_transform(label)

        label = self.label_encode(label)
        return image, label

# transform = v2.Compose([
#     v2.ConvertImageDtype(torch.float32),
#     v2.Resize((360, 480)), 
#     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# target_transform = v2.Resize((360, 480), interpolation=InterpolationMode.NEAREST)

# train_dataset = CamVidDataset(img_dir='CamVid/train_images', label_dir='CamVid/train_labels', class_dict_dir='CamVid/class_dict.csv', transform=transform, target_transform=target_transform)
# test_dataset = CamVidDataset(img_dir='CamVid/test_images', label_dir='CamVid/test_labels', class_dict_dir='CamVid/class_dict.csv', transform=transform, target_transform=target_transform)

# result = torch.zeros(32)
# for i in range(train_dataset.__len__()):
#     _, tensor = train_dataset.__getitem__(i)
#     flattened = tensor.flatten()
#     counts = torch.bincount(flattened, minlength=32)
#     result = result + counts
# print(result.to(torch.int64))