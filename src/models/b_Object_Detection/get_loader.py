import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from PIL import Image
import os

class FlickrDataset(Dataset):
    def __init__(self, root_folder, annotation_file, transform):
        self.root_folder = root_folder
        self.annotations = annotation_file
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id, caption = self.annotations[index]
        img = Image.open(os.path.join(self.root_folder, img_id)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, caption

def get_loader(root_folder, annotation_file, transform, num_workers=0, batch_size=32, shuffle=True):
    dataset = FlickrDataset(root_folder, annotation_file, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return loader, dataset
