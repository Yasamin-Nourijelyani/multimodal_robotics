import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import re
import ast


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

    vocab, model = load_your_model_and_vocab_somehow(device)  
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

    comment = "Image description: [{'name': 'blue', 'keypoint': {'x': 296, 'y': 552}}, {'name': 'yellow', 'keypoint': {'x': 331, 'y': 551}}, {'name': 'green', 'keypoint': {'x': 336, 'y': 512}}, {'name': 'orange', 'keypoint': {'x': 323, 'y': 485}}, {'name': 'blue', 'keypoint': {'x': 310, 'y': 454}}, {'name': 'yellow', 'keypoint': {'x': 352, 'y': 469}}, {'name': 'blue', 'keypoint': {'x': 395, 'y': 518}}, {'name': 'yellow', 'keypoint': {'x': 420, 'y': 548}}, {'name': 'orange', 'keypoint': {'x': 428, 'y': 444}}, {'name': 'blue', 'keypoint': {'x': 430, 'y': 486}}, {'name': 'blue', 'keypoint': {'x': 459, 'y': 457}}, {'name': 'blue', 'keypoint': {'x': 496, 'y': 459}}, {'name': 'yellow', 'keypoint': {'x': 481, 'y': 480}}, {'name': 'green', 'keypoint': {'x': 484, 'y': 500}}, {'name': 'green', 'keypoint': {'x': 498, 'y': 546}}]"
    prompt = prompt_template(comment)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = lm_model.generate(input_ids=inputs["input_ids"].to(device), max_new_tokens=280)
    output_text = tokenizer.batch_decode(outputs)[0]
    dict_string = re.search(r"\[/INST\] (.*?)\[/s\]", text).group(1)
    extracted_dict = ast.literal_eval(dict_string)
    extracted_dict
    print(extracted_dict)


if __name__ == "__main__":
    image_path = "../../../data/coord_text_images_random/images/synthetic_image_1.png"
    main(image_path)
