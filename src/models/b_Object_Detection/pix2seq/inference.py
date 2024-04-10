import torch
from transformers import top_k_top_p_filtering
from config import CFG
from config import CFG
from tokenizer import KeypointTokenizer
from model import Encoder, Decoder, EncoderDecoder
import pandas as pd
import albumentations as A
import cv2
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import os
import config

GT_COLOR = (0, 255, 0) # Green
PRED_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def generate(model, x, tokenizer, max_len=50, top_k=0, top_p=1):
    x = x.to(CFG.device)
    batch_preds = torch.ones(x.size(0), 1).fill_(tokenizer.BOS_code).long().to(CFG.device)
    confs = []
    
    if top_k != 0 or top_p != 1:
        sample = lambda preds: torch.softmax(preds, dim=-1).multinomial(num_samples=1).view(-1, 1)
    else:
        sample = lambda preds: torch.softmax(preds, dim=-1).argmax(dim=-1).view(-1, 1)
        
    with torch.no_grad():
        for i in range(max_len):
            preds = model.predict(x, batch_preds)
            ## If top_k and top_p are set to default, the following line does nothing
            preds = top_k_top_p_filtering(preds, top_k=top_k, top_p=top_p)
            if i % 4 == 0:
                confs_ = torch.softmax(preds, dim=-1).sort(axis=-1, descending=True)[0][:, 0].cpu()
                confs.append(confs_)
            preds = sample(preds)
            batch_preds = torch.cat([batch_preds, preds], dim=1)
    
    return batch_preds.cpu(), confs


def postprocess(batch_preds, batch_confs, tokenizer):
    EOS_idxs = (batch_preds == tokenizer.EOS_code).float().argmax(dim=-1)
    ## sanity check
    invalid_idxs = ((EOS_idxs - 1) % 5 != 0).nonzero().view(-1)
    EOS_idxs[invalid_idxs] = 0
    
    all_keypoints = []
    all_labels = []
    all_confs = []
    for i, EOS_idx in enumerate(EOS_idxs.tolist()):
        if EOS_idx == 0:
            all_keypoints.append([])
            all_labels.append([])
            all_confs.append([])
            continue
        labels, keypoints = tokenizer.decode(batch_preds[i, :EOS_idx+1])
        confs = [round(batch_confs[j][i].item(), 3) for j in range(len(keypoints))]

       
        all_keypoints.append(keypoints)
        all_labels.append(labels)
        all_confs.append(confs)
        
    return all_keypoints, all_labels, all_confs

class VOCDatasetTest(torch.utils.data.Dataset):
    def __init__(self, img_paths, size):
        self.img_paths = img_paths
        self.transforms = A.Compose([A.Resize(size, size), A.Normalize()])

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img = cv2.imread(img_path)[..., ::-1]

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        img = torch.FloatTensor(img).permute(2, 0, 1)

        return img

    def __len__(self):
        return len(self.img_paths)


def visualize_keypoints(img, keypoints, color=GT_COLOR, thickness=2):
    """Visualizes keypoints on the image"""
    assert isinstance(color, tuple) and len(color) in [3, 4], f"Invalid color value: {color}"
    for keypoint in keypoints:
        x, y = int(keypoint[0]), int(keypoint[1])
        cv2.circle(img, (x, y), radius=1, color=color, thickness=thickness)
    return img

def visualize(image, keypoints, category_ids, category_id_to_name, color=PRED_COLOR, show=True):
    img = image.copy()
    for keypoints, category_id in zip(keypoints, category_ids):
        img = visualize_keypoints(img, keypoints, color)
    if show:
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)
        plt.show()
    return img

if __name__ == "__main__":


    test_csv_file_path = 'data/test_imgloc_caption.csv'

    valid_df = pd.read_csv(test_csv_file_path)


    classes = sorted(valid_df['names'].unique())
    cls2id = {cls_name: i for i, cls_name in enumerate(classes)}
    id2cls = {i: cls_name for i, cls_name in enumerate(classes)}
    num_classes = len(classes)  # num of unique labels
    num_bins = CFG.num_bins     # num of bins for quantization
    width = CFG.img_size       
    height = CFG.img_size       
    max_len = CFG.max_len       # maximum sequence length

    tokenizer = KeypointTokenizer(num_classes=num_classes, num_bins=num_bins,
                                width=width, height=height, max_len=max_len)

    CFG.pad_idx = tokenizer.PAD_code




    encoder = Encoder(model_name=CFG.model_name, pretrained=True, out_dim=256)
    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                    encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(encoder, decoder)
    model.to(CFG.device)

    msg = model.load_state_dict(torch.load('./best_valid_loss.pth', map_location=CFG.device))
    print(msg)
    model.eval()

    img_paths = """synthetic_image_10651.png"""
    img_paths = ["./data/coord_text_images_random/images/" + path for path in img_paths.split(" ")]

    test_dataset = VOCDatasetTest(img_paths, size=CFG.img_size)
    test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=len(img_paths), shuffle=False, num_workers=0)


    all_keypoints = []
    all_labels = []
    all_confs = []

    with torch.no_grad():
        for x in tqdm(test_loader):
            batch_preds, batch_confs = generate(
                model, x, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1)
            keypoints, labels, confs = postprocess(
                batch_preds, batch_confs, tokenizer)
            all_keypoints.append(keypoints)
            all_labels.append(labels)
            all_confs.append(confs)

    for i, (keypoints, labels, confs) in enumerate(zip(all_keypoints, all_labels, all_confs)):
        img_path = img_paths[i]
        img = cv2.imread(img_path)[..., ::-1]
        img = cv2.resize(img, (CFG.img_size, CFG.img_size))
        img = visualize(img, keypoints, labels, id2cls, color=PRED_COLOR, show=False)

        cv2.imwrite("results/" + img_path.split("/")[-1], img[..., ::-1])

    text_output_path = "results/detection_output.txt"
    with open(text_output_path, 'w') as file:
        for i, (keypoints, labels, confs) in enumerate(zip(all_keypoints, all_labels, all_confs)):


            for keypoint, label, conf in zip(keypoints, labels, confs):
                print("label__________", labels)
                print("label__________", keypoints)
                print("label__________", confs)
                file.write(f"Image {i}, Label: {id2cls[label]}, Confidence: {conf}, Keypoints: {keypoint}\n")
