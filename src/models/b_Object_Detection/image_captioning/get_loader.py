import os
import pandas as pd
import torch
import spacy # for tokenizer
from torch.nn.utils.rnn import pad_sequence # to pad the batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import json

# python -m spacy download en
spacy_eng = spacy.load("en_core_web_sm")
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
                



class CoordDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab: Vocabulary, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = self.load_annotations_file(captions_file)
        self.transform = transform

        self.imgs = self.df["image_path"]
        self.captions = self.df["caption"]

        # init vocab and build
        self.vocab = vocab
        self.vocab.build_vocabulary(self.captions.tolist())

    def load_annotations_file(self, annotations_file):
        data = {
            'image_path': [],
            'caption': []
        }
        with open(annotations_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                data['image_path'].append(entry['image_path'])
                data['caption'].append(entry['caption'])
        return pd.DataFrame(data)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)
    
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets
    

# only 1 vocab for training and test data
def build_vocab_from_training_data(train_annotation_file, freq_threshold=5):
    # load captions
    train_df = pd.read_json(train_annotation_file, lines=True)
    captions = train_df["caption"].tolist()
    
    # build vocabulary
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(captions)
    
    return vocab



def get_data_loaders(train_annotations_file, test_annotations_file, root_folder, transform, batch_size=32, num_workers=4, shuffle=True, pin_memory=True):

    vocab = build_vocab_from_training_data(train_annotations_file)

    train_dataset = CoordDataset(root_folder, train_annotations_file, vocab, transform=transform)
    test_dataset = CoordDataset(root_folder, test_annotations_file, vocab, transform=transform)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=vocab.stoi["<PAD>"])
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=vocab.stoi["<PAD>"])
    )
    
    return train_loader, test_loader, train_dataset, test_dataset



def main():
    transform = transforms.Compose(
        [
            transforms.Resize((756, 660)),
            transforms.ToTensor(),
        ]
    )
    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(
        train_annotations_file="../../../data/train_test_data/train_imgloc_caption.jsonl",
        test_annotations_file="../../../data/train_test_data/test_imgloc_caption.jsonl",
        root_folder="../../../data/coord_text_images_random/images",
        transform=transform
    )

if __name__ == "__main__":
    main()
