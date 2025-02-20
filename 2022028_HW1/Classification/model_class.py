import wandb
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
import torchvision.models as models
from dataset_class import CustomImageDataset, get_dataloaders
from sklearn.metrics import f1_score

class ConvNet(nn.Module):
    def __init__(self, num_classes: int=10):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(14*14*128, num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
    
class ResNet18(nn.Module):
    def __init__(self, num_classes: int=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights='DEFAULT')
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.model(x)
    
def train(train_dataloader, val_dataloader, device, model, criterion, optimizer, config, model_name):
    best_val_loss = float("inf")
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        num_train_samples = 0
        num_val_samples = 0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curr_batch_size = images.size(0)
            train_loss += loss.item() * curr_batch_size
            num_train_samples += curr_batch_size

        average_train_loss = train_loss/num_train_samples

        model.eval()
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                curr_batch_size = images.size(0)
                val_loss += loss.item() * curr_batch_size
                num_val_samples += curr_batch_size

        average_val_loss = val_loss/num_val_samples

        wandb.log({"epoch": epoch + 1, "train_loss": average_train_loss, "val_loss": average_val_loss})
        print(f"EPOCH: {epoch+1} | TRAIN LOSS: {average_train_loss} | VAL LOSS: {average_val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"weights/{model_name}.pth")
    print(f"weights saved at weights/{model_name}.pth")

def test(val_dataloader, device, weight_dir, model_class, class_names):
    model = model_class().to(device)
    model.load_state_dict(torch.load(weight_dir))
    model.eval()
    
    preds = []
    y_true = []
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred_batch = torch.max(outputs.data, 1)
            preds.extend(pred_batch.tolist())
            y_true.extend(labels.tolist())
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None, preds=preds, y_true=y_true, class_names=class_names)})
        print("confusion matrix saved to wandb")
        accuracy = sum(np.array(preds) == np.array(y_true)) / len(y_true)
        f1 = f1_score(y_true, preds, average="weighted")
        print(f"ACCURACY: {accuracy:.4f}")
        print(f"F1 SCORE: {f1:.4f}")

def model_pipeline(model_name, config):
    with wandb.init(project=model_name, config=config):
        config = wandb.config

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        class_names = ['amur_leopard', 'amur_tiger', 'birds', 'black_bear', 'brown_bear', 'dog', 'roe_deer', 'sika_deer', 'wild_boar', 'people' ]
        img_dir = 'russian-wildlife-dataset/Cropped_final'  

        transform = v2.Compose([
            v2.ConvertImageDtype(torch.float32),
            v2.Resize((224, 224)), 
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_aug = v2.Compose([
            v2.ConvertImageDtype(torch.float32),
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if model_name == "convnet":
            dataset = CustomImageDataset(img_dir, class_names, transform=transform)
            model_class = ConvNet
            weight_dir = "weights/convnet.pth"
        elif model_name == "resnet":
            dataset = CustomImageDataset(img_dir, class_names, transform=transform)
            model_class = ResNet18
            weight_dir = "weights/resnet.pth"

        elif model_name == "resnet_aug":
            dataset = CustomImageDataset(img_dir, class_names, transform=transform_aug)
            model_class = ResNet18
            weight_dir = "weights/resnet_aug.pth"
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        train_dataloader, val_dataloader = get_dataloaders(dataset=dataset, batch_size=config.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_class().to(device)        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        train(train_dataloader, val_dataloader, device, model, criterion, optimizer, config, model_name)

        print(f"Evaluation of {model_name} on VAL SET:")
        test(val_dataloader, device, weight_dir, model_class, class_names)

wandb.login()
config = dict(
    epochs=10,
    batch_size=32,
    learning_rate=0.001,
    dataset="russian-wildlife-dataset",
    architecture="CNN",
    seed=2022028)

# model_pipeline(model_name="convnet", config=config)
# model_pipeline(model_name="resnet", config=config)
model_pipeline(model_name="resnet_aug", config=config)