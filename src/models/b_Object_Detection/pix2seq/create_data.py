import pandas as pd
import json
import numpy as np
from PIL import Image, ImageDraw
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from test_train_csv import create_df


# generate data (random blocks and caption), split to test and train, create dataframe    

def draw_3d_block(draw, origin, block_size, block_depth, color, color_name):
    x, y = origin
    draw.polygon([(x, y), (x + block_size, y), 
                  (x + block_size, y + block_size), (x, y + block_size)], fill=color)
    draw.polygon([(x, y), (x + block_depth, y - block_depth), 
                  (x + block_size + block_depth, y - block_depth), (x + block_size, y)], fill=color)
    draw.polygon([(x + block_size, y), (x + block_size + block_depth, y - block_depth), 
                  (x + block_size + block_depth, y + block_size - block_depth), 
                  (x + block_size, y + block_size)], fill=color)
    return {
        'name': color_name,
        'keypoint': {'x': x, 'y': y},
        
    }




def create_json():

    colors = [
        ((255, 255, 0), 'yellow'),
        ((255, 165, 0), 'orange'),
        ((0, 0, 255), 'blue'),
        ((0, 128, 0), 'green')
    ]

    num_images = 20000
    blocks_per_image = 10
    block_size = 15
    block_depth = 5
    target_size = (384, 384)

    out_dir_images = 'data/coord_text_images_random/images/'
    out_dir_caption = 'data/coord_text_images_random/'

    if not os.path.exists(out_dir_images):
        os.makedirs(out_dir_images)


    text_file_content = ["image$caption"]

    for img_index in range(num_images):
        block_metadata = []

        img = Image.new('RGB', target_size, color='white')
        draw = ImageDraw.Draw(img)

        head_radius = 30
        head_center = (target_size[0] // 2, head_radius + 10)
        head_bbox = (head_center[0] - head_radius, head_center[1] - head_radius,
                    head_center[0] + head_radius, head_center[1] + head_radius)
        draw.ellipse(head_bbox, fill=(0, 0, 0))  # black

        body_radius = 60
        body_center = (head_center[0], head_center[1] + head_radius + body_radius)
        body_bbox = (body_center[0] - body_radius, body_center[1] - body_radius,
                    body_center[0] + body_radius, body_center[1])
        draw.pieslice(body_bbox, 180, 360, fill=(0, 0, 0))

        for block_index in range(blocks_per_image):
            x = np.random.randint(0, target_size[0] - block_size - block_depth)
            y = np.random.randint(target_size[1] - 200, target_size[1] - block_size)
            
            color, color_name = colors[np.random.randint(0, len(colors))]
            
            block_data = draw_3d_block(draw, (x, y), block_size, block_depth, color, color_name)
            block_metadata.append(block_data)

        caption = f"{block_metadata}"
        text_file_content.append(f"synthetic_image_{img_index + 1}.png${caption}")

        img = img.resize(target_size, Image.Resampling.LANCZOS)


        img_path = f"{out_dir_images}synthetic_image_{img_index + 1}.png"


        img.save(img_path)

        text_file_path = f"{out_dir_caption}captions.txt"
        with open(text_file_path, 'w') as f:
            for line in text_file_content:
                f.write(line + "\n")




    # split the data to test and train
    data = {
        'image_path': [],
        'caption': []
    }
    with open(text_file_path, 'r') as f:
        next(f)  # Skip the header line
        for line in f:
            image_path, caption = line.strip().split('$', 1)
            img_pth_full = out_dir_images + image_path
            data['image_path'].append(img_pth_full)
            data['caption'].append(caption)
    df = pd.DataFrame(data)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)


    train_df = train_df.to_dict(orient='records')
    with open("data/train_imgloc_caption.jsonl", 'w') as f:
        for line in train_df:
            f.write(json.dumps(line) + "\n")


    test_df = test_df.to_dict(orient='records')
    with open("data/test_imgloc_caption.jsonl", 'w') as f:
        for line in test_df:
            f.write(json.dumps(line) + "\n")







if __name__ == "__main__":

    create_json()

    train_file_path = 'data/train_imgloc_caption.jsonl'  
    test_file_path = 'data/test_imgloc_caption.jsonl' 

    train_csv_file_path = 'data/train_imgloc_caption.csv'  
    test_csv_file_path = 'data/test_imgloc_caption.csv'

    create_df(train_file_path, train_csv_file_path)
    create_df(test_file_path, test_csv_file_path)
   