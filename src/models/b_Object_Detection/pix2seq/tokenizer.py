import gc
import os
import cv2
import math
import random
from glob import glob
import numpy as np
import pandas as pd
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A
import xml.etree.ElementTree as ET
from sklearn.model_selection import StratifiedGroupKFold

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import cv2
from torch.utils.data import Dataset
import timm
from timm.models.layers import trunc_normal_
import transformers
from transformers import top_k_top_p_filtering
from transformers import get_linear_schedule_with_warmup
import albumentations as A
from config import CFG

def get_transform_train():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Resize(height=660, width=756), 
        A.Normalize(),
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))  

def get_transform_valid():
    return A.Compose([
        A.Resize(height=660, width=756),  
        A.Normalize(),
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))  

class KeypointDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.ids = df['id'].unique()
        self.df = df
        self.transforms = transforms

    def __getitem__(self, idx):
        sample = self.df[self.df['id'] == self.ids[idx]]
        img_path = sample['img_path'].values[0]
        
        img = cv2.imread(img_path)[..., ::-1]  
        keypoints = sample[['x', 'y']].values  
        labels = sample['label'].values

        if self.transforms is not None:
            transformed = self.transforms(image=img, keypoints=keypoints, labels=labels)
            img = transformed['image']
            keypoints = transformed['keypoints']
            labels = transformed['labels']

        img = torch.FloatTensor(img).permute(2, 0, 1)
        keypoints = torch.FloatTensor(keypoints)
        
        return img, labels, keypoints

    def __len__(self):
        return len(self.ids)



class KeypointTokenizer:
    def __init__(self, num_classes: int, num_bins: int, width: int, height: int, max_len=500):
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.width = width
        self.height = height
        self.max_len = max_len

        self.BOS_code = num_classes + num_bins * 2  # start token
        self.EOS_code = self.BOS_code + 1           # end token
        self.PAD_code = self.EOS_code + 1           # padding token

        self.vocab_size = num_classes + num_bins * 2 + 3

    def quantize(self, x: np.array):
        return (x * (self.num_bins - 1)).astype('int')

    def dequantize(self, x: np.array):
        return x.astype('float32') / (self.num_bins - 1)

    def __call__(self, labels: list, keypoints: list):
        assert len(labels) == len(keypoints) // 2, "Each label should correspond to a pair of keypoints (x, y)"
        
        keypoints = np.array(keypoints).reshape(-1, 2)  
        labels = np.array(labels)

        labels += self.num_bins * 2
        labels = labels.astype('int')[:self.max_len]

        keypoints[:, 0] = keypoints[:, 0] / self.width  # normalize x
        keypoints[:, 1] = keypoints[:, 1] / self.height # normalize y
        keypoints = self.quantize(keypoints)[:self.max_len]

        tokenized = [self.BOS_code]  # start with BOS token
        for label, (x, y) in zip(labels, keypoints):
            tokenized.extend([x, y, label])
        tokenized.append(self.EOS_code)  # end with EOS token

        tokenized = tokenized[:self.max_len]
        while len(tokenized) < self.max_len:
            tokenized.append(self.PAD_code)  
        return tokenized

    def decode(self, tokens: torch.tensor):
        tokens = tokens.numpy()
        mask = tokens != self.PAD_code
        tokens = tokens[mask]
        tokens = tokens[1:-1]  # eemove BOS and EOS toks

        labels = []
        keypoints = []
        for i in range(0, len(tokens), 3):
            x, y, label = tokens[i:i+3]
            keypoints.extend([x, y])
            labels.append(label - self.num_bins * 2)

        labels = np.array(labels)
        keypoints = np.array(keypoints).reshape(-1, 2)
        keypoints = self.dequantize(keypoints)

        # go back to original scale
        keypoints[:, 0] = keypoints[:, 0] * self.width
        keypoints[:, 1] = keypoints[:, 1] * self.height

        return labels, keypoints

if __name__ == "__main__":
    
    num_classes = len(classes)  # Number of unique labels
    num_bins = CFG.num_bins     # Number of bins for quantization
    width = CFG.img_size        # Image width
    height = CFG.img_size       # Image height (assuming square images for simplicity)
    max_len = CFG.max_len       # Maximum sequence length

    tokenizer = KeypointTokenizer(num_classes=num_classes, num_bins=num_bins,
                                width=width, height=height, max_len=max_len)

    # For a dictionary, use CFG['num_bins'], CFG['img_size'], etc.

    # Update the CFG with the PAD token index used by the tokenizer
    CFG.pad_idx = tokenizer.PAD_code
