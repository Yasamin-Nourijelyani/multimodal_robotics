import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import os



block_df = pd.read_csv('data/block_location.csv')

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

color_map = {
    'yellow': (255, 255, 0),
    'orange': (255, 165, 0),
    'blue': (0, 0, 255),
    'green': (0, 128, 0)
}

num_images = block_df['img_config'].nunique()
blocks_per_image = 10
block_size = 25
block_depth = 10

out_dir = 'data/coord_text_images_non_random/images/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

block_metadata = []

text_file_content = ["image$caption"]

for img_config in block_df['img_config'].unique():
    img_df = block_df[block_df['img_config'] == img_config]      


    width, height = 756, 660
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    block_metadata = []


    # Draw head (oval)
    head_radius = 70
    head_center = (width // 2, head_radius + 10)
    head_bbox = (head_center[0] - head_radius, head_center[1] - head_radius,
                 head_center[0] + head_radius, head_center[1] + head_radius)
    draw.ellipse(head_bbox, fill=(0, 0, 0))  # Black

    # Draw body (semicircle)
    body_radius = 140
    body_center = (head_center[0], head_center[1] + head_radius + body_radius)
    body_bbox = (body_center[0] - body_radius, body_center[1] - body_radius,
                 body_center[0] + body_radius, body_center[1])
    draw.pieslice(body_bbox, 180, 360, fill=(0, 0, 0))

    for _, row in img_df.iterrows():
            x, y = row['x'], row['y']
            color_name = row['color']
            color = color_map[color_name]

            block_data = draw_3d_block(draw, (x, y), block_size, block_depth, color, color_name)
            block_metadata.append(block_data)


    caption = f"{block_metadata}"
    text_file_content.append(f"synthetic_image_{img_config}.png${caption}")

    img = img.resize((384, 384), Image.Resampling.LANCZOS)

    img_path = f"{out_dir}synthetic_image_{img_config}.png"
    img.save(img_path)

text_file_path = f"{out_dir}captions.txt"
with open(text_file_path, 'w') as f:
    for line in text_file_content:
        f.write(line + "\n")

