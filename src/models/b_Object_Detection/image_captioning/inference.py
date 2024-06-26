import torch
import torchvision.transforms as transforms
from PIL import Image
from models.b_Object_Detection.image_captioning.model import CNNtoRNN
from models.b_Object_Detection.image_captioning.utils import load_checkpoint
from get_loader import get_data_loaders, Vocabulary  


def load_model(device, checkpoint_path, vocab_size, embed_size=256, hidden_size=256, num_layers=1):
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    if 'optimize' in checkpoint:  
        optimizer.load_state_dict(checkpoint['optimize'])
    model.eval()
    return model

def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((756, 660)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # add batch dimension

def generate_caption(image_tensor, model, device, vocab):
    model.eval() 
    image_tensor = image_tensor.to(device)
    feature = model.encoderCNN(image_tensor)
    captions = model.caption_image(feature, vocab)

    filtered_indices = [idx for idx in captions if idx not in (vocab.stoi.get("<SOS>", -1), vocab.stoi.get("<EOS>", -1), vocab.stoi.get("<PAD>", -1))]
    sentence = ' '.join([vocab.itos.get(idx, "<UNK>") for idx in filtered_indices])

    return sentence

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    transform = transforms.Compose(
        [
            transforms.Resize((756, 660)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(
        train_annotations_file="../../../data/train_imgloc_caption.jsonl",
        test_annotations_file="../../../data/test_imgloc_caption.jsonl",
        root_folder="../../../data",
        transform=transform
    )
    vocab = train_dataset.vocab
    vocab_size = len(vocab)
    checkpoint_path = "results/checkpoint.pth"
    
    model = load_model(device, checkpoint_path, vocab_size)
  
    captions = []
    for imgs, _ in train_loader:
        for img_tensor in imgs:
            caption = generate_caption(img_tensor.unsqueeze(0), model, device, vocab)  #  image passed in tensor directly
            captions.append(caption)
    print("done")
    return captions

if __name__ == "__main__":
    captions = main()
    print(captions)
