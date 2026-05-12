
# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms.v2 import ToDtype, Compose, Resize, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, ElasticTransform
from torchvision.transforms import v2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision import datasets

np.random.seed(100)


# %%
class GalaxyZooDataset(Dataset):
    def __init__(self, img_labels, img_dir, img_size=(64, 64), transform=None, 
                target_transform=Compose([
                    list,
                    torch.tensor,
                    ToDtype(torch.float32)
                ]), 
                loader=Image.open):
        
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.img_size = img_size


        labels = [self.img_labels[idx, 1:] for idx in range(len(self.img_labels))]
        labels = [self.target_transform(label) for label in labels]
        self.labels = np.array(labels)


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(int(self.img_labels[idx, 0])) + ".jpg")
        image = self.loader(img_path)
        label = self.img_labels[idx, 1:]

        image = self.transform(image)
        label = self.target_transform(label)
        return image, label
    
        



# %% Load data

img_labels = pd.read_csv("./data/training_solutions_rev1/training_solutions_rev1.csv")
img_labels = np.array(img_labels)
np.random.shuffle(img_labels)
train_img_labels, val_img_labels, test_img_labels = img_labels[:len(img_labels)*7//10], img_labels[len(img_labels)*7//10:len(img_labels)*8//10], img_labels[len(img_labels)*8//10:]

print(test_img_labels[0, 0])
# print(len(img_labels))
# print(len(train_img_labels) + len(val_img_labels) + len(test_img_labels))

# %%
# label_slice = list(range(37)[:3])
# print(label_slice)

cropsize = 212
size = 64
img_size=(size, size)


train_dataset = GalaxyZooDataset(
    img_labels=train_img_labels,
    img_dir="./data/images_training_rev1",
    img_size=img_size,
    transform=Compose([
                    v2.RandomAffine(degrees=360, translate=(0.01, 0.01), scale=(0.9, 1)),
                    v2.CenterCrop(cropsize),
                    Resize(img_size),
                    # v2.RandomApply(torch.nn.ModuleList([v2.RandomResizedCrop(img_size, scale=(0.9, 0.9))]), p=0.5),
                    # v2.RandomRotation(degrees=360),
                    RandomHorizontalFlip(),
                    # RandomVerticalFlip(),
                    # v2.RandomApply(torch.nn.ModuleList([ElasticTransform(alpha=50)]), p=0.2),
                    # v2.RandomApply(torch.nn.ModuleList([v2.GaussianBlur(kernel_size=3)]), p=0.2),
                    # v2.RandomApply(torch.nn.ModuleList([v2.GaussianNoise(mean=0.0, sigma=0.05)]), p=0.2),
                    ToTensor(),
                    # ToDtype(torch.float32)
                ])
)
val_dataset = GalaxyZooDataset(
    img_labels=val_img_labels,
    img_dir="./data/images_training_rev1",
    img_size=img_size,
    transform=Compose([
                    v2.CenterCrop(cropsize),
                    Resize(img_size),
                    ToTensor(),
                    # ToDtype(torch.float32)
                ]),
)
test_dataset = GalaxyZooDataset(
    img_labels=test_img_labels,
    img_dir="./data/images_training_rev1",
    img_size=img_size,
    transform=Compose([
                    v2.CenterCrop(cropsize),
                    Resize(img_size),
                    ToTensor(),
                    # ToDtype(torch.float32)
                ])
)


# %%

# print(len(train_dataset))
# print(train_dataset[0][0].shape)
# print(train_dataset[0][1].shape)

# %%
for i in range(5):
    im = train_dataset[i][0].permute(1,2,0)
    plt.imshow(im)
    plt.show()

# %% Showcasing class imbalance
mean_labels = np.mean(train_dataset.labels, 0)
print(mean_labels)

possible_cost_weights = [float(np.log(1/l)) for l in mean_labels]
print(possible_cost_weights)
# %% 
print(np.max(train_dataset.labels, 0))


# %% 

# train, val, test = random_split(dataset, [0.7, 0.1, 0.2])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# %% Define Galaxy Zoo specific output function to scale outputs mathing the questions asked by galaxy zoo
# import theano
# import theano.tensor as T
class GalaxyZooOutputFunction(nn.Module): # Inspired by https://github.com/benanne/kaggle-galaxies/blob/master/doc/documentation.pdf
    def __init__(self, name="GalaxyZooOutputFunction"):
        super().__init__()
        self.name = name
        self.slices = [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 9), slice(9, 13), slice(13, 15),
                                slice(15, 18), slice(18, 25), slice(25, 28), slice(28, 31), slice(31, 37)]


    def forward(self, x):
        # print("---------------------------------------")
        y = nn.Sigmoid()(x)
        
        z = y.clone()
        for question_slice in self.slices:
            z[:, question_slice] = torch.div(y[:, question_slice], torch.sum(y[:, question_slice], dim=1, keepdim=True) + 1e-12)

        w = z.clone()
        q1 = w[:, self.slices[0]]
        q2 = w[:, self.slices[1]]
        q3 = w[:, self.slices[2]]
        q4 = w[:, self.slices[3]]
        q5 = w[:, self.slices[4]]
        q6 = w[:, self.slices[5]]
        q7 = w[:, self.slices[6]]
        q8 = w[:, self.slices[7]]
        q9 = w[:, self.slices[8]]
        q10 = w[:, self.slices[9]]
        q11 = w[:, self.slices[10]]

        q1_scaled = q1
        q2_scaled = torch.mul(q2, q1_scaled[:, 1].unsqueeze(1))
        q3_scaled = torch.mul(q3, q2_scaled[:, 1].unsqueeze(1))
        q4_scaled = torch.mul(q4, q2_scaled[:, 1].unsqueeze(1))
        q5_scaled = torch.mul(q5, q2_scaled[:, 1].unsqueeze(1))
        q6_scaled = q6
        q7_scaled = torch.mul(q7, q1_scaled[:, 0].unsqueeze(1))
        q8_scaled = torch.mul(q8, q6_scaled[:, 0].unsqueeze(1))
        q9_scaled = torch.mul(q9, q2_scaled[:, 0].unsqueeze(1))
        q10_scaled = torch.mul(q10, q4_scaled[:, 0].unsqueeze(1))
        q11_scaled = torch.mul(q11, q4_scaled[:, 0].unsqueeze(1))

        out = torch.cat([q1_scaled, q2_scaled, q3_scaled, q4_scaled, q5_scaled, q6_scaled, q7_scaled, q8_scaled, q9_scaled, q10_scaled, q11_scaled], dim=1)

        return out

# %%
x = train_dataset[0][1]
x1 = train_dataset[1][1]
xx = torch.stack([x, x1])
print(xx)
# print(xx.shape)
print(GalaxyZooOutputFunction()(xx))

# %% Define the model

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class GalaxyModel(nn.Module):
    def __init__(self, name="GalaxyModel"):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(5*5*128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 37),
            # nn.Linear(128, 3),

            GalaxyZooOutputFunction()
            # nn.Sigmoid()
        )
        self.name = name
        # self.loss_function = RMSELoss
        # self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return self.model(x)
    

class CommonSenseModel(nn.Module):
    def __init__(self, means, name="CommonSenseModel"):
        self.means = means
        self.name = name
        super().__init__()

    def forward(self, x):
        return torch.tensor(self.means).repeat(x.shape[0], 1)
    
commonSense = CommonSenseModel(mean_labels)

galaxy_nn = GalaxyModel().to(device)
print(galaxy_nn.model)

# %%
class WRMSELoss(nn.Module):
    def __init__(
        self,
        weight = torch.ones(37),
        # reduction: str = "mean",
    ) -> None:
        super().__init__()
        # self.reduction = reduction
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = self.weight.repeat(input.shape[0],1)
        # print(input.shape)
        # print(target.shape)
        # print(weights.shape)
        weighted_squared_errors = nn.functional.mse_loss(input, target, reduction="none", weight=weights)
        wrmse = torch.sqrt(torch.sum(weighted_squared_errors) / torch.sum(weights))
        wrse_per_label = torch.sqrt(torch.div(torch.sum(weighted_squared_errors, 0), torch.sum(weights, 0)))
        return wrmse, wrse_per_label

wrmse_loss_fun = WRMSELoss()

# %% Define train and test loops

def RMSELoss(yhat,y):
    criterion = nn.MSELoss()
    return torch.sqrt(criterion(yhat,y))


# %%

# loss_fun = nn.CrossEntropyLoss()
def test_loop(dataloader, model, mode="Test", loss_fun=RMSELoss):
    print("---------------------------------------")
    print(f"Running {mode} for {model.name}")
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    losses = []
    w_losses = []
    w_losses_per_label = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fun(pred, y)
            losses.append(loss.item())

            w_loss, label_losses = wrmse_loss_fun(pred, y)
            w_losses.append(w_loss.item())
            w_losses_per_label.append(label_losses)

    test_loss = np.mean(losses)
    print(f"{mode} Error: \n  Avg loss: {test_loss:>8f} \n")
    
    test_w_loss = np.mean(w_losses)
    print(f"{mode} Error: \n  Avg w loss: {test_w_loss:>8f} \n")

    label_loss = np.mean(w_losses_per_label, 0)
    print(f"{mode} Error: \n  Avg loss per label: {label_loss} \n")
    print("---------------------------------------")
    return test_loss

def train_loop(training_loader, validation_loader, model, epochs=10, loss_fun=RMSELoss):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_val_losses = []
    epoch_train_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        size = len(training_loader.dataset)
        model.train()
        losses = []
        for batch, (x, y) in enumerate(training_loader):
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fun(pred, y)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            
            epoch_train_losses.append(loss.item())  
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end="\r")
        print(f"Avg training loss: {np.mean(losses):>7f}")

        val_loss = test_loop(validation_loader, model, mode="Validation")
        epoch_val_losses.append(val_loss)

# %% Training loop
train_loop(train_loader, val_loader, galaxy_nn, epochs=20)

# %% Evaluate the model
test_loop(test_loader, commonSense)
test_loop(test_loader, galaxy_nn)

# %%
# print(commonSense([1,2,3]))

# %%
pred = galaxy_nn(test_loader.dataset[0][0].unsqueeze(0))
print(pred)
print(test_loader.dataset[0][1])
# %%
