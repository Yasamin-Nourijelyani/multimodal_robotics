import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import re
import ast


def load_model_and_vocab(device, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    stoi = checkpoint['vocab']
    vocab = Vocabulary()
    vocab.stoi = stoi
    vocab.itos = {v: k for k, v in stoi.items()}
    
    model = CNNtoRNN(embed_size=256, hidden_size=256, vocab_size=len(vocab), num_layers=1).to(device)
    model.load_state_dict(checkpoint['state_dict'])

    return vocab, model

def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((756, 660)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def main(image_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "results/checkpoint.pth"

    vocab, model = load_model_and_vocab(device, checkpoint_path)  
    image_tensor = transform_image(image_path)
    
    caption = generate_caption(image_tensor, model, device, vocab)
    print("Generated caption:", caption)

    # Instructions string for robotics task

    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = PeftConfig.from_pretrained("nourijel/robotics_finetuned_text_perception")
    lm_model = PeftModel.from_pretrained(model_name, config=config)

    intstructions_string = f""" Output only the keypoint location of the block corresponding to following instruction. Instructions are from the perspective of the black figure. Instruction:Pick up the blue block on your left, which is the second from the left nearest you."""

    prompt_template = lambda comment: f'''[INST] {intstructions_string} \n{comment} \n[/INST]'''

    prompt = prompt_template(caption)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = lm_model.generate(input_ids=inputs["input_ids"].to(device), max_new_tokens=280)
    output_text = tokenizer.batch_decode(outputs)[0]
    dict_string = re.search(r"\[/INST\] (.*?)\[/s\]", text).group(1)
    extracted_dict = ast.literal_eval(dict_string)
    extracted_dict
    print(extracted_dict)


if __name__ == "__main__":
    image_path = "../../../data/coord_text_images_non_random/images/synthetic_image_1.png"
    main(image_path)
