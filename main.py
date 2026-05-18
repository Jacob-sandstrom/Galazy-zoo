
# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms.v2 import ToDtype, Compose, Resize, ToTensor, RandomHorizontalFlip, RandomAffine, CenterCrop
from PIL import Image
from torch.utils.data import Dataset, DataLoader


np.random.seed(100)
torch.manual_seed(100)

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

        # Store all labels for easy access
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
    
        
# %% Load data and split into train, validation and test sets
img_labels = pd.read_csv("./data/training_solutions_rev1/training_solutions_rev1.csv")
img_labels = np.array(img_labels)
np.random.shuffle(img_labels)
train_img_labels, val_img_labels, test_img_labels = img_labels[:len(img_labels)*7//10], img_labels[len(img_labels)*7//10:len(img_labels)*8//10], img_labels[len(img_labels)*8//10:]


# %%
cropsize = 153
size = 51
img_size=(size, size)


train_dataset = GalaxyZooDataset(
    img_labels=train_img_labels,
    img_dir="./data/images_training_rev1",
    img_size=img_size,
    transform=Compose([
                    RandomAffine(degrees=360, translate=(0.01, 0.01), scale=(1, 1.1)),
                    CenterCrop(cropsize),
                    Resize(img_size),
                    RandomHorizontalFlip(),
                    ToTensor()
                ])
)
val_dataset = GalaxyZooDataset(
    img_labels=val_img_labels,
    img_dir="./data/images_training_rev1",
    img_size=img_size,
    transform=Compose([
                    CenterCrop(cropsize),
                    Resize(img_size),
                    ToTensor()
                ]),
)
test_dataset = GalaxyZooDataset(
    img_labels=test_img_labels,
    img_dir="./data/images_training_rev1",
    img_size=img_size,
    transform=Compose([
                    CenterCrop(cropsize),
                    Resize(img_size),
                    ToTensor()
                ])
)

# %% plot some images to check the dataset
# for i in range(5):
#     im = train_dataset[i][0].permute(1,2,0)
#     plt.imshow(im)
#     plt.show()


# %% Define data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Only one batch for validation and test to get the average loss over the whole set
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


# %% Define Galaxy Zoo specific output function to scale outputs mathing the questions asked by galaxy zoo
class GalaxyZooOutputFunction(nn.Module): # Inspired by https://github.com/benanne/kaggle-galaxies/blob/master/doc/documentation.pdf
    def __init__(self, name="GalaxyZooOutputFunction"):
        super().__init__()
        self.name = name
        self.slices = [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 9), slice(9, 13), slice(13, 15),
                                slice(15, 18), slice(18, 25), slice(25, 28), slice(28, 31), slice(31, 37)]

    def forward(self, x):
        # y = nn.Sigmoid()(x)
        # y = nn.ReLU()(x)
        y = x.clone()
        
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

        out = torch.cat([q1_scaled, q2_scaled, q3_scaled, q4_scaled, q5_scaled, 
                         q6_scaled, q7_scaled, q8_scaled, q9_scaled, q10_scaled, q11_scaled], dim=1)
        return out

# %% Make sure the output function works as intended
# x = train_dataset[0][1]
# x1 = train_dataset[1][1]
# xx = torch.stack([x, x1])
# print(xx)
# print()
# print(GalaxyZooOutputFunction()(xx))
# pred = galaxy_nn(train_dataset[0][0].unsqueeze(0))
# print(x)
# print(pred)
# _, l = WRMSELoss()(pred, x.unsqueeze(0))
# print(l)
# print()

# %% Define the model
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class GalaxyModel(nn.Module):
    def __init__(self, name="GalaxyModel"):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=6, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            # nn.Dropout(0.2),
            nn.Linear(4*4*128, 2048),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(2048, 37),
            nn.ReLU(),


            GalaxyZooOutputFunction()
            # nn.Sigmoid()
        )
        self.name = name

    def forward(self, x):
        return self.model(x)
    

class CommonSenseModel(nn.Module):
    def __init__(self, means, name="CommonSenseModel"):
        self.means = means
        self.name = name
        super().__init__()

    def forward(self, x):
        return torch.tensor(self.means).repeat(x.shape[0], 1)
    
mean_labels = np.mean(train_dataset.labels, 0)
var_labels = np.var(train_dataset.labels, 0)
commonSense = CommonSenseModel(mean_labels)

galaxy_nn = GalaxyModel().to(device)
print(galaxy_nn.model)

# %%
class WRMSELoss(nn.Module):
    def __init__(
        self,
        weight = torch.ones(37),
    ) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = self.weight.repeat(input.shape[0],1)
        weighted_squared_errors = nn.functional.mse_loss(input, target, reduction="none", weight=weights)
        wrmse = torch.sqrt(torch.sum(weighted_squared_errors) / torch.sum(weights))

        wrse_per_label = torch.sqrt(torch.div(torch.sum(weighted_squared_errors, 0), torch.sum(weights, 0)))
        return wrmse, wrse_per_label

# wrmse_loss_fun = WRMSELoss()

# %% Define train and test loops
def test_loop(dataloader, model, mode="Test", loss_fun=WRMSELoss()):
    unweighted_loss_fun = WRMSELoss()
    # print("---------------------------------------")
    print(f"Running {mode} for {model.name} ----------")
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    losses = []
    unweighted_losses = []
    losses_per_label = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss, label_losses = loss_fun(pred, y)
            losses.append(loss.item())
            losses_per_label.append(label_losses)

            unweighted_loss, _ = unweighted_loss_fun(pred, y)
            unweighted_losses.append(unweighted_loss.item())

    test_loss = np.mean(losses)
    print(f"{mode} Error: \n  Avg loss: {test_loss:>8f} \n")
    
    test_uw_loss = np.mean(unweighted_losses)
    print(f"{mode} Error: \n  Avg unweighted loss: {test_uw_loss:>8f} \n")

    label_loss = np.mean(losses_per_label, 0)
    print(f"{mode} Error: \n  Avg loss per label: {label_loss} \n")
    print("##########################################")
    return test_loss, label_loss

def train_loop(training_loader, validation_loader, model, epochs, loss_fun=WRMSELoss()):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    val_losses = []
    val_label_losses = []
    train_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}    ----------------------------------")
        size = len(training_loader.dataset)
        model.train()
        losses = []
        for batch, (x, y) in enumerate(training_loader):
            x, y = x.to(device), y.to(device)

            pred = model(x)
            # loss = loss_fun(pred, y)
            loss, _ = loss_fun(pred, y)
            
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end="\r")
        mean_loss = np.mean(losses)
        print(f"Avg training loss: {mean_loss:>7f}           ")
        train_losses.append(mean_loss)

        val_loss, label_loss = test_loop(validation_loader, model, mode="Validation", loss_fun=loss_fun)
        val_losses.append(val_loss)
        val_label_losses.append(label_loss)
    return val_losses, val_label_losses, train_losses


# %% Training loop

# # Weights based on class imbalance and classification accuracy of model without weighting
# weight = torch.tensor([1,1,10, 5,1, 5,1, 1,5, 10,5,5,10, 1,1, 1,1,10, 5,10,10,5,5,5,10, 1,10,1, 5,5,1, 10,1,10,10,10,5])

# Weights calculated from validation dataset on model trained without weights as w = 1/(recall+0.1) where recall is the proportion of samples with label > 0.7 that have prediction > 0.5. The 0.1 is added to avoid extremely high weights for classes with very low recall.
weight = torch.tensor([0.9511135816574097, 0.9314120411872864, 10.0, 0.9519004225730896, 0.9369822144508362, 1.0865874290466309, 0.9840983748435974, 0.9977966547012329, 1.5113871097564697, 10.0, 1.1022576093673706, 1.0969793796539307, 10.0, 1.0945684909820557, 0.9138110280036926, 0.9713574647903442, 1.09375, 2.307692289352417, 1.0447760820388794, 10.0, 10.0, 10.0, 10.0, 1.919191837310791, 10.0, 1.0646387338638306, 10.0, 0.9090908765792847, 2.307692289352417, 1.428571343421936, 1.27516770362854, 10.0, 1.1170213222503662, 10.0, 10.0, 10.0, 10.0])
train_loss_fun = WRMSELoss(weight=weight)

# train_loss_fun = WRMSELoss()
# train_loss_fun = nn.CrossEntropyLoss()
# train_loss_fun = nn.L1Loss()
# loss = WRMSELoss(weight=)

model_path = "./model/galaxy_nn.pth"
continue_training = True
if continue_training:
    galaxy_nn.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


# %%
val_losses, val_label_losses, train_losses = train_loop(train_loader, val_loader, galaxy_nn, epochs=5, loss_fun=train_loss_fun)

model_path = "./model/galaxy_nn_weighted_pretrained.pth"
torch.save(galaxy_nn.model.state_dict(), model_path)


# %% Evaluate the model
test_loss_common_sense, label_loss_common_sense = test_loop(test_loader, commonSense)
test_loss, label_loss = test_loop(test_loader, galaxy_nn, loss_fun=train_loss_fun)

label_improvement = np.subtract(label_loss_common_sense, label_loss)

# %%

print(len(val_losses))

plt.plot(val_losses, label="Validation Loss")
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# %%
# label_slice = slice(0, 6)
label_slice = [0,1,3,4,6,7,8]
label_slice = slice(0, 37)
val_label_losses = np.array(val_label_losses)
plt.plot(val_label_losses[:, label_slice])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss per Label")
# plt.legend(np.array(["Class1.1","Class1.2","Class1.3","Class2.1","Class2.2","Class3.1","Class3.2","Class4.1","Class4.2","Class5.1","Class5.2","Class5.3","Class5.4","Class6.1","Class6.2","Class7.1","Class7.2","Class7.3","Class8.1","Class8.2","Class8.3","Class8.4","Class8.5","Class8.6","Class8.7","Class9.1","Class9.2","Class9.3","Class10.1","Class10.2","Class10.3","Class11.1","Class11.2","Class11.3","Class11.4","Class11.5","Class11.6"])[label_slice])
plt.show()



# -------------------------------------------------------------------------------------------------------
#
#
#   Plotting:
#
#
# -------------------------------------------------------------------------------------------------------


# %% Test the model on a single example
# galaxy_nn.model.eval()
# sample_im, sample_label = train_loader.dataset[0]

# pred = galaxy_nn(sample_im.unsqueeze(0))
# print(pred)
# print(sample_label)



# %%
class_names = np.array(["Class1.1","Class1.2","Class1.3","Class2.1","Class2.2","Class3.1","Class3.2","Class4.1","Class4.2","Class5.1","Class5.2","Class5.3","Class5.4","Class6.1","Class6.2","Class7.1","Class7.2","Class7.3","Class8.1","Class8.2","Class8.3","Class8.4","Class8.5","Class8.6","Class8.7","Class9.1","Class9.2","Class9.3","Class10.1","Class10.2","Class10.3","Class11.1","Class11.2","Class11.3","Class11.4","Class11.5","Class11.6"])

plt.bar(class_names, label_loss_common_sense, alpha=0.5, label="Common sense model loss per class")
plt.bar(class_names, label_loss, width=0.5, alpha=0.8, label="trained model loss per class")

plt.ylabel("RMSE Loss")
plt.title("Loss per class for trained model and common sense model")
plt.xticks(rotation=90)
plt.xlabel("Class")

plt.legend()
plt.show()

# %% Showcasing class imbalance

# cutoff = 0.5
# using_dataset = train_dataset
# data_size = len(using_dataset)
# classes = {}
# for i in range(37):
#     classes[class_names[i]] = np.where(using_dataset.labels[:, i] > cutoff)

# for i in range(37):
#     print(f"{class_names[i]}: {len(classes[class_names[i]][0])} samples")



# plt.bar(class_names, [len(classes[class_name][0])/data_size for class_name in class_names], label=f"Low confidence samples (label > {cutoff})", alpha=1)
cutoff = 0.7
using_dataset = test_dataset
data_size = len(using_dataset)
classes = {}
for i in range(37):
    classes[class_names[i]] = np.where(using_dataset.labels[:, i] > cutoff)

for i in range(37):
    print(f"{class_names[i]}: {len(classes[class_names[i]][0])} samples")



plt.bar(class_names, [len(classes[class_name][0])/data_size for class_name in class_names], label=f"High confidence samples (label > {cutoff})", width=0.5)
plt.xticks(rotation=90)
plt.xlabel("Class")
plt.ylabel(f"Proportion of samples belonging to class ")
plt.title(f"Class distribution in training set. N samples: {data_size}")
# plt.ylim(0, 0.1)
# plt.yscale("log")
plt.legend()
plt.show()


# %%
plt.bar(class_names, [len(classes[class_name][0])/data_size for class_name in class_names], width=1, label="Class distribution in training set", alpha=1)
plt.bar(class_names, label_loss_common_sense, alpha=0.5, label="Common sense model loss per class")
plt.bar(class_names, label_loss, width=0.5, alpha=0.8, label="trained model loss per class")

# plt.ylabel("RMSE Loss")
# plt.title("Loss per class for trained model and common sense model")
plt.xticks(rotation=90)
plt.xlabel("Class")

plt.legend()
plt.show()

# %%
galaxy_nn.model.eval()
recalls = []
for class_i in range(37):
# class_i = 0
    predictions = []
    for i in classes[class_names[class_i]][0]:
        im, label = using_dataset[i]
        # plt.imshow(im.permute(1,2,0))
        # plt.title(f"Sample from class {class_names[class_i]} with label {label[class_i]:.2f}")
        # plt.show()
        pred = galaxy_nn(using_dataset[i][0].unsqueeze(0).to(device))
        predictions.append(pred[0][class_i].item())
        # print(pred)
        # print(f"Sample from class {class_names[class_i]} with label {label[class_i]:.2f} and prediction {pred[0][class_i].item():.2f}")

    n_correct = np.where(np.array(predictions) > 0.5)[0].size
    recalls.append(n_correct/(len(predictions)+1e-12))
    print(f"{class_names[class_i]}: {n_correct} / {len(predictions)}. Recall: {n_correct/(len(predictions)+1e-12):.2f}")

# %%
plt.figure(figsize=(10,5))
ax = plt.subplot()
ax.bar(class_names, label_loss, width=0.5, alpha=0.8, label="Model loss")
ax.bar(class_names, recalls, label="Recall", alpha=0.5)
ax.legend(loc="upper right")
ax.set_xticks(range(len(class_names)), labels=class_names, rotation=90)
for xtick, color in zip(ax.get_xticklabels(), ["black" if len(classes[class_name][0]) > 0 else "red" for class_name in class_names]):
    xtick.set_color(color)
ax.set_xlabel("Class")
ax.set_ylabel("Recall (proportion of samples with prediction > 0.5)\n RMSE model Loss")
ax.set_title(f"Accuracy (recall) of trained model on test data with high class confidence(label > {cutoff})")
plt.show()


# %%
# print(label_loss)

# print(torch.div(torch.ones(37), torch.tensor(label_loss)))

# print(torch.div(torch.ones(37), torch.add(torch.tensor(recalls), torch.ones(37)*0.1)))
w = torch.div(torch.ones(37), torch.add(torch.tensor(recalls), torch.ones(37)*0.1))
w[2] = 10
print(list([float(weight) for weight in w]))
# %%
