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
import re
import ast 
from src.models.b_Object_Detection.pix2seq.inference import VOCDatasetTest
from tqdm import tqdm
from PIL import Image, ImageDraw


def pix2seq(img_path, test_csv_file_path):

    valid_df = pd.read_csv(test_csv_file_path)
    classes = sorted(valid_df['names'].unique())
    num_classes = len(classes)
    cls2id = {cls_name: i for i, cls_name in enumerate(classes)}
    id2cls = {i: cls_name for i, cls_name in enumerate(classes)}
    num_bins = CFG.num_bins     # num of bins for quantization
    width = CFG.img_size       
    height = CFG.img_size       
    max_len = CFG.max_len    

    tokenizer = KeypointTokenizer(num_classes=num_classes, num_bins=num_bins,
                            width=width, height=height, max_len=max_len)

    CFG.pad_idx = tokenizer.PAD_code 

    # Initialize the model
    encoder = Encoder(model_name=CFG.model_name, pretrained=True, out_dim=256)
    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                    encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(encoder, decoder)
    model.to(CFG.device)

    msg = model.load_state_dict(torch.load('src/models/b_Object_Detection/pix2seq/best_valid_loss.pth', map_location=CFG.device))
    #print(msg)
    model.eval()



    img_paths = [img_path]

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
        img = visualize(img, keypoints, labels, id2cls, color=CFG.PRED_COLOR, show=False)

        cv2.imwrite("results/" + img_path.split("/")[-1], img[..., ::-1])

    text_output_path = "src/models/b_Object_Detection/pix2seq/results/detection_output.txt"
    with open(text_output_path, 'w') as file:
        for i, (keypoints, labels, confs) in enumerate(zip(all_keypoints, all_labels, all_confs)):


            for keypoint, label, conf in zip(keypoints[0], labels[0], confs[0]):

                file.write(f"Image {i}, Label: {id2cls[label]}, Confidence: {conf}, Keypoints: {keypoint}\n")


    # Define a list to hold the image descriptions
    image_descriptions = []

    # Regular expression to match the lines and capture the relevant data
    line_pattern = re.compile(
        r"Image \d+, Label: (\w+), Confidence: [\d.]+, Keypoints: \[([\d.]+) ([\d.]+)\]"
    )

    # Open and read the text file and convert it to correct format
    with open(text_output_path, "r") as file:
        for line in file:
            match = line_pattern.match(line)
            if match:
                label, x, y = match.groups()
                # convert x and y to float and round them 
                keypoint = {'x': round(float(x)), 'y': round(float(y))}
                # append the extracted information to the image descriptions list
                image_descriptions.append({'name': label, 'keypoint': keypoint})

    # Example output
    #print(f"Image description_______: {image_descriptions}")
    return image_descriptions



def llm(text, instruction):


    # Run python -m models.a_LM.download_and_save_model to download trained model once


    model_directory = "src/models/a_LM/model_directory"
    model = AutoModelForCausalLM.from_pretrained(model_directory, device_map="cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(model_directory)


    intstructions_string = f""" Output only the keypoint location of the block corresponding to following 
                                instruction. Instructions are from the perspective of the black figure. 
                                Instruction:{instruction}"""

    prompt_template = lambda text: f'''[INST] {intstructions_string} \n{text} \n[/INST]'''
    prompt = prompt_template(text)


    model.eval()


    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

    text = tokenizer.batch_decode(outputs)[0]
    #print(text)
    answer = text.split("[/INST]")[1]
    answer = answer.split("}")[0]
    answer = answer + "}"
    #dict_string = re.search(r"\[/INST\] {(.*?)}\[/", text).group(1)
    extracted_dict = ast.literal_eval(answer.strip())
    return extracted_dict
    
def plot_keypoint(img_path, coords, keypoint_img_path):


    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)

    radius = 2

    left_up_point = (coords['x'] - radius, coords['y'] - radius)
    right_down_point = (coords['x'] + radius, coords['y'] + radius)
    draw.ellipse([left_up_point, right_down_point], fill='red')


    img.save(keypoint_img_path)
    print(f"Image with keypoint saved in location: {keypoint_img_path}")




if __name__ == "__main__":

    # to be changed for inference:
    # locate an image from the test dataset: 'models/b_Object_Detection/pix2seq/data/test_imgloc_caption.jsonl' 
    instruction = "Locate the green blocks that is near the back, just under two blue blocks."
    img_path = """data/coord_text_images_non_random/images/synthetic_image_3.png"""
    # where to save image after plotting keypoint
    keypoint_img_path = """data/modified_images/synthetic_image_3.png"""
#_________________________________________________________________________________________________________
    # path for vocab - do not change
    test_csv_file_path = 'src/models/b_Object_Detection/pix2seq/data/test_imgloc_caption.csv'
    # running the model pix2seq > llm
    text = pix2seq(img_path, test_csv_file_path)
    extracted_dict = llm(text, instruction)
    print(extracted_dict)
    #final results plotted
    plot_keypoint(img_path, extracted_dict, keypoint_img_path)