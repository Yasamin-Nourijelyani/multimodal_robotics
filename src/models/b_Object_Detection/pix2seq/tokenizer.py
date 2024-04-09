import cv2
import numpy as np
from functools import partial
import albumentations as A
import torch
from torch.nn.utils.rnn import pad_sequence
import cv2
from torch.utils.data import Dataset
from config import CFG


def get_transform_train():
    """image preprocess"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Resize(CFG.img_size, CFG.img_size), 
        A.Normalize(),
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))  

def get_transform_valid():
    """image preprocess"""
    return A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),  
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
    """For tokenizing keypoints """
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
        """
        toekns: torch.LongTensor with shape [L]
        """
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
    

def collate_fn(batch, max_len, pad_idx):
    """
    if max_len:
        the sequences will all be padded to that length
    """
    image_batch, seq_batch = [], []
    for image, seq in batch:
        image_batch.append(image)
        seq_batch.append(seq)

    seq_batch = pad_sequence(
        seq_batch, padding_value=pad_idx, batch_first=True)
    if max_len:
        pad = torch.ones(seq_batch.size(0), max_len -
                         seq_batch.size(1)).fill_(pad_idx).long()
        seq_batch = torch.cat([seq_batch, pad], dim=1)
    image_batch = torch.stack(image_batch)
    return image_batch, seq_batch

def get_loaders(train_df, valid_df, tokenizer, img_size, batch_size, max_len, pad_idx, num_workers=2):

    train_ds = KeypointDataset(train_df, transforms=get_transform_train(), tokenizer=tokenizer)

    trainloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_ds = KeypointDataset(valid_df, transforms=get_transform_valid(), tokenizer=tokenizer)

    validloader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=2,
        pin_memory=True,
    )

    return trainloader, validloader

