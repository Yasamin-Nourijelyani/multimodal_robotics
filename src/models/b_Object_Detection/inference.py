import torch
from PIL import Image
from torchvision import transforms
from model import CNNtoRNN
from get_loader import get_loader


transform = transforms.Compose(
        [
            transforms.Resize((756, 660)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

train_loader, dataset = get_loader(
        root_folder = "../../../data/coord_text_images_test/images",
        annotation_file="../../../data/coord_text_images_test/captions.txt",
        transform=transform,
        num_workers = 2
    )

embed_size = 256
hidden_size = 256
vocab_size = len(dataset.vocab)
num_layers = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def load_trained_model(model_path, device):
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    # Load the trained model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model



def generate_caption(image_tensor, model, vocab, max_length=50):
    """
    Generate a caption for the given image tensor.
    
    Args:
    - image_tensor: Tensor of shape (1, C, H, W) representing a preprocessed image.
    - model: Trained captioning model.
    - vocab: Vocabulary object with stoi (string to index) and itos (index to string) methods.
    - max_length: Maximum length of the generated caption.
    
    Returns:
    - A string representing the generated caption.
    """
    model.eval()  
    start_token = vocab.stoi["<SOS>"]
    end_token = vocab.stoi["<EOS>"]
    
    caption = []
    
    words = torch.tensor([start_token], device=image_tensor.device).unsqueeze(0)
    
    with torch.no_grad():
        features = model.encoderCNN(image_tensor).unsqueeze(0)
        
        states = None
        
        for _ in range(max_length):
            hiddens, states = model.decoderRNN(words, features, states)
            predicted = hiddens.argmax(dim=2)  
            words = predicted
            
            predicted_word_idx = predicted.item()
            caption.append(vocab.itos[predicted_word_idx])
            
            if predicted_word_idx == end_token:
                break
    

    caption = ' '.join(caption[:-1])  # exclude the end token in the final caption
    
    return caption



def generate_caption_from_image_path(image_path, model, transform, vocab, device):
    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate a caption
    model.eval()
    with torch.no_grad():
        caption = generate_caption(image_tensor, model, vocab)
    
    return caption

# Define your transformations
transform = transforms.Compose([
    transforms.Resize((756, 660)),
    transforms.CenterCrop((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Path to your trained model checkpoint
model_path = "results/checkpoint.pth"

# Assuming device, vocab are already defined
model = load_trained_model(model_path, device)

# Example usage
image_path = "../../../data/coord_text_images_test/images/synthetic_image_1.png"
caption = generate_caption_from_image_path(image_path, model, transform, dataset.vocab, device)
print("Generated caption:", caption)
