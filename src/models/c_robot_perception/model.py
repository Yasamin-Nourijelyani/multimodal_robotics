import torch
import pandas as pd
from src.models.b_Object_Detection.pix2seq.tokenizer import KeypointTokenizer
from src.models.b_Object_Detection.pix2seq.model import Encoder, Decoder, EncoderDecoder
import albumentations as A
import cv2
from matplotlib import pyplot as plt
from src.models.b_Object_Detection.pix2seq.inference import generate, postprocess, visualize
from src.models.b_Object_Detection.pix2seq.config import CFG
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re
import ast 




def pix2seq(img_path):

    num_classes = CFG.num_classes  # num of unique labels
    num_bins = CFG.num_bins     # num of bins for quantization
    width = CFG.img_size       
    height = CFG.img_size       
    max_len = CFG.max_len     

    # Initialize the model
    encoder = Encoder(model_name=CFG.model_name, pretrained=True, out_dim=256)
    decoder = Decoder(vocab_size=CFG.vocab_size, encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(encoder, decoder)
    model.load_state_dict(torch.load('../b_Object_Detection/pix2seq/best_valid_loss.pth', map_location=CFG.device))
    model.eval().to(CFG.device)

    tokenizer = KeypointTokenizer(num_classes=num_classes, num_bins=num_bins,
                                width=width, height=height, max_len=max_len)

    CFG.pad_idx = tokenizer.PAD_code


    # Image preprocessing
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_transformed = transform(image=img)['image']
    img_transformed = torch.FloatTensor(img_transformed).permute(2, 0, 1).unsqueeze(0).to(CFG.device)

    # Inference
    with torch.no_grad():
        batch_preds, batch_confs = generate(model, img_transformed, CFG.tokenizer, max_len=CFG.max_len, top_k=CFG.top_k, top_p=CFG.top_p)
        keypoints, labels, confs = postprocess(batch_preds, batch_confs, CFG.tokenizer)

    # Visualization
    img = cv2.resize(img, (CFG.img_size, CFG.img_size))
    img = visualize(img, keypoints[0], labels[0], CFG.id2cls, CFG.PRED_COLOR, show=True)

    # Optionally, write keypoints information to a text file
    text_output_path = "single_image_detection_output.txt"
    with open(text_output_path, 'w') as file:
        for keypoint, label, conf in zip(keypoints[0], labels[0], confs[0]):
            file.write(f"Label: {CFG.id2cls[label]}, Confidence: {conf}, Keypoints: {keypoint}\n")


    # Define a list to hold the image descriptions
    image_descriptions = []

    # Regular expression to match the lines and capture the relevant data
    line_pattern = re.compile(
        r"Image \d+, Label: (\w+), Confidence: [\d.]+, Keypoints: \[([\d.]+) ([\d.]+)\]"
    )

    # Open and read the text file and convert it to correct format
    with open("single_image_detection_output.txt", "r") as file:
        for line in file:
            match = line_pattern.match(line)
            if match:
                label, x, y = match.groups()
                # convert x and y to float and round them 
                keypoint = {'x': round(float(x)), 'y': round(float(y))}
                # append the extracted information to the image descriptions list
                image_descriptions.append({'name': label, 'keypoint': keypoint})

    # Example output
    print(f"Image description: {image_descriptions}")
    return image_descriptions



def llm(text):



    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="main")

    model = PeftModel.from_pretrained(model, "nourijel/robotics_finetuned_text_perception")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


    intstructions_string = f""" Output only the keypoint location of the block corresponding to following instruction. Instructions are from the perspective of the black figure. Instruction:Pick up the blue block on your left, which is the second from the left nearest you."""

    prompt_template = lambda text: f'''[INST] {intstructions_string} \n{text} \n[/INST]'''

    prompt = prompt_template(text)
    #print(prompt)


    model.eval()


    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

    text = tokenizer.batch_decode(outputs)[0]
    print(text)
    dict_string = re.search(r"\[/INST\] (.*?)\[/s\]", text).group(1)
    extracted_dict = ast.literal_eval(dict_string)
    return extracted_dict
    


if __name__ == "__main__":




    img_path = "../b_Object_Detection/pix2seq/data/coord_text_images_random/images/synthetic_image_10651.png"
    text = pix2seq(img_path)
    extracted_dict = llm(text)
    print(extracted_dict)