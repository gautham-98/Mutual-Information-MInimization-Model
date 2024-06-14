from torch.utils.data.dataset import Dataset
import torch
from PIL import Image
import cv2 as cv
from Dataset.Utils import pad_resize_image, apply_clahe
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import config.Load_Parameter

# Function to display images
def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

class DatasetOneLabel(Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target = self.dataset[index]
        return example, target

    def __len__(self):
        return len(self.dataset)

class DatasetTwoLabels(Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target1, target2 = self.dataset[index]
        return example, target1, target2

    def __len__(self):
        return len(self.dataset)

class DatasetMultiLabels(Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, *targets = self.dataset[index]
        return example, *targets

    def __len__(self):
        return len(self.dataset)
    
class ImageLabelDataset(Dataset):
    def __init__(self, _df):
        self.dataset = _df

    def __getitem__(self, index):
        img_path, *labels = self.dataset.iloc[index]
        full_img_path = '/data/public/chexpert/' + img_path
        # example = Image.open(full_img_path).convert('RGB')
        example = cv.imread(full_img_path)
        example = self.transform(example)
        example = example.cuda()
        labels = [torch.tensor(label).long() for label in labels]
        return example, *labels

    def __len__(self):
        return len(self.dataset)
    
    def transform(self, example):
        example = cv.cvtColor(example, cv.COLOR_BGR2GRAY)
        example = apply_clahe(example)
        example = Image.fromarray(example)
        example = pad_resize_image(example)
        if config.Load_Parameter.params.selectFeatureEncoder == 0:
            # for custom feature encoder 1x300x300
            example = np.array(example)
            example = torch.FloatTensor(example)
            example = example.unsqueeze(0)
            normalise = transforms.Normalize(mean=0.5, std=1)
            example = normalise(example)
        
        elif config.Load_Parameter.params.selectFeatureEncoder == 1:
            # convert to RGB and normalise for densenet-121 3x96x96
            preprocess = transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
                    ])
            example = preprocess(example)

            augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.3),
                transforms.RandomApply([transforms.RandomAffine(degrees=15, translate=(0, 0), scale=(0.9, 1.1), shear=10)],p=0.3),
                transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.3),
            ])
            example = augmentation(example)
        return example






