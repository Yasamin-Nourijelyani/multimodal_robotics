import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNNtoRNN
from utils import load_checkpoint
from get_loader import get_data_loaders, Vocabulary  


def load_model(device, checkpoint_path, vocab_size, embed_size=256, hidden_size=256, num_layers=1):
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
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

def generate_caption(image_path, model, device, vocab):
    model.eval() 
    image_tensor = transform_image(image_path).to(device)
    feature = model.cnn(image_tensor)
    sampled_ids = model.rnn.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()  
    
    # IDs to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.itos[word_id]
        sampled_caption.append(word)
        if word == "<EOS>":
            break
    sentence = ' '.join(sampled_caption[1:-1])  #skipping <SOS> and <EOS>
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

    _, test_loader, _, test_dataset = get_data_loaders(
        train_annotations_file="../../../data/train_test_data/train_imgloc_caption.jsonl",
        test_annotations_file="../../../data/train_test_data/test_imgloc_caption.jsonl",
        root_folder="../../../data/coord_text_images_random/images",
        transform=transform
    )
    vocab = test_dataset.vocab
    vocab_size = len(vocab)
    checkpoint_path = "results/checkpoint.pth"
    
    model = load_model(device, checkpoint_path, vocab_size)
  
    captions = []
    for image_paths, _ in test_loader:
        for image_path in image_paths:
            caption = generate_caption(image_path, model, device, vocab)
            captions.append((image_path, caption))
    print("done")
    return captions

if __name__ == "__main__":
    captions = main()
    print(captions)
