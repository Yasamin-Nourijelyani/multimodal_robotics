import torch
import random

def save_checkpoint(checkpoint, filename):
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint.get('step', 0)
    return step


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
    model.eval()  # Put model in evaluation mode
    
    # Start token
    start_token = vocab.stoi["<SOS>"]
    words = [start_token]
    
    with torch.no_grad():  # No need to track gradients
        for _ in range(max_length):
            captions_tensor = torch.LongTensor(words).unsqueeze(0).to(image_tensor.device)
            predictions = model(image_tensor, captions_tensor)
            
            # Predict the next word token ID (with the highest probability)
            predicted_id = predictions.argmax(1)[-1].item()
            words.append(predicted_id)
            
            # End if <EOS> is generated
            if predicted_id == vocab.stoi["<EOS>"]:
                break
    
    # Convert word IDs back to strings
    generated_caption = ' '.join([vocab.itos[idx] for idx in words[1:-1]])  # Skip <SOS> and <EOS> in the final caption
    
    return generated_caption


def print_examples(model, device, dataset):
    model.eval()
    for _ in range(10):  # Print 10 examples
        idx = random.randint(0, len(dataset)-1)
        img, _ = dataset[idx]
        img = img.unsqueeze(0).to(device)
        caption = generate_caption(img, model, dataset.vocab)  
        print(f"Generated caption: {caption}")
    model.train()
