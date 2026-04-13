
# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder, DatasetFolder, VisionDataset
from torchvision.transforms import Compose, Resize, ToTensor, PILToTensor
from pathlib import Path
from typing import Any, Callable, cast, Optional, Union
from PIL import Image


# %% Define the data transformations

transform_img = Compose([
    Resize((64, 64)),
    ToTensor()
])

# %%
images = os.listdir("./data/images_training_rev1")
images = [img for img in images if img.endswith(".jpg")]

print(len(images))



# %% Load data

img_transform = Compose([
    Resize((64, 64)),
    PILToTensor()
])

img_tensors = [img_transform(Image.open(f"./data/images_training_rev1/{img}")) for img in images[:1000]]
# img_tensors = list(map(lambda x: (x-x.min())/(x.max()-x.min()), img_tensors))     # Normalize the images to [0, 1]

# %%

print(torch.max(img_tensors[0]))
print(torch.min(img_tensors[0]))

print(img_tensors[0].shape)

plt.imshow(img_tensors[0].permute(1, 2, 0))
plt.show()


# %% Define the model

n = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU()

)

