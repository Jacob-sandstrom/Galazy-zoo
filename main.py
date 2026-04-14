
# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, PILToTensor
from torchvision.transforms.v2 import Lambda, ToDtype
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision import datasets

# %%
class GalaxyZooDataset(Dataset):
    def __init__(self, annotations_file, img_dir, 
                 transform=Compose([
                    Resize((64, 64)),
                    ToTensor(),
                    # ToDtype(torch.float32)
                ]), 
                target_transform=Compose([
                    list,
                    torch.tensor,
                    ToDtype(torch.float32)
                ]), 
                loader=Image.open):
        
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.img_size = (64, 64)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]) + ".jpg")
        image = self.loader(img_path)
        label = self.img_labels.iloc[idx, 1:]

        image = self.transform(image)
        label = self.target_transform(label)
        return image, label




# %% Load data
dataset = GalaxyZooDataset(
    annotations_file="./data/training_solutions_rev1/training_solutions_rev1.csv",
    img_dir="./data/images_training_rev1"
)


# %%

print(len(dataset))
print(dataset[0][0].shape)
print(dataset[0][1].shape)



# %% 

train, val, test = random_split(dataset, [0.7, 0.1, 0.2])

train_loader = DataLoader(train, batch_size=128, shuffle=True)
val_loader = DataLoader(val, batch_size=128, shuffle=False)
test_loader = DataLoader(test, batch_size=128, shuffle=False)

# %% Define the model

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class GalaxyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(64*64//4//4*32, 128),
            nn.ReLU(),

            nn.Linear(128, 37),
            nn.Sigmoid()
        )
        # self.loss_function = RMSELoss
        # self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return self.model(x)
    
galaxy_nn = GalaxyModel().to(device)
print(galaxy_nn.model)

# %% Define train and test loops
# print(train_loader.dataset[0][0].unsqueeze(0))
# galaxy_nn.forward(train_loader.dataset[0][0].unsqueeze(0))

def RMSELoss(yhat,y):
    criterion = nn.MSELoss()
    return torch.sqrt(criterion(yhat,y))

loss_fun = nn.CrossEntropyLoss()
def test_loop(dataloader, model, mode="Test", loss_fun=RMSELoss):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    # test_loss = 0
    losses = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fun(pred, y)
            # test_loss += loss.item()
            losses.append(loss.item())


    test_loss = np.mean(losses)
    print(f"{mode} Error: \n  Avg loss: {test_loss:>8f} \n")
    return test_loss

def train_loop(training_loader, validation_loader, model, epochs=10, loss_fun=RMSELoss):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_val_losses = []
    epoch_train_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        size = len(training_loader.dataset)
        model.train()
        for batch, (x, y) in enumerate(training_loader):
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fun(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            
            epoch_train_losses.append(loss.item())  
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        val_loss = test_loop(validation_loader, model, mode="Validation")
        epoch_val_losses.append(val_loss)

# %% Training loop
train_loop(train_loader, val_loader, galaxy_nn, epochs=10)

# %% Evaluate the model
test_loop(test_loader, galaxy_nn)



# %%
pred = galaxy_nn(test_loader.dataset[0][0].unsqueeze(0))
print(pred)
print(test_loader.dataset[0][1])
# %%
