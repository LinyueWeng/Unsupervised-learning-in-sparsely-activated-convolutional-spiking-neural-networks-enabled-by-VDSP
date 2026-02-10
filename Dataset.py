import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

class ImageDataset(datasets):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0,1)
    ])

    def __init__(self,images, labels=None, num_classes=None):
        self.images = images
        if labels is None:
            labels = np.zeros(len(images))
        if num_classes is None:
            num_classes = len(np.unique(labels))
        self.label = labels
        self.num_classes = num_classes
        self.image_size = images.shape[1:]

    def __getitem__(self, index):
        image = ImageDataset.transform(self.images[index])
        label = self.label[index]
        return image, label

