import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import os
import uuid

# Function to draw 3D block
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
        'id': None,
        'keypoint': {'x': x, 'y': y},
        'name': color_name,
    }

# Specific colors and their names
colors = [
    ((255, 255, 0), 'yellow'),
    ((255, 165, 0), 'orange'),
    ((0, 0, 255), 'blue'),
    ((0, 128, 0), 'green')
]

# Define the number of images and blocks per image
num_images = 1000
blocks_per_image = 10
block_size = 25
block_depth = 10

# Define output directory for images
out_dir = 'images/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Placeholder for block metadata
block_metadata = []

# Placeholder for text file content
text_file_content = ["image$caption"]

for img_index in range(num_images):
      # Placeholder for block metadata
    block_metadata = []

    # Placeholder for text file content
    # Create a new image with specified dimensions
    width, height = 756, 660
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

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

    # Draw 3D blocks
    for block_index in range(blocks_per_image):
        # Generate random coordinates for block placement
        x = np.random.randint(0, width - block_size - block_depth)
        y = np.random.randint(height - 300, height - block_size)
        
        # Randomly pick a color
        color, color_name = colors[np.random.randint(0, len(colors))]
        
        # Draw 3D block and get metadata
        block_data = draw_3d_block(draw, (x, y), block_size, block_depth, color, color_name)
        block_data.update({
            'id': str(uuid.uuid4())
        })
        block_metadata.append(block_data)

      # Append to text file content
    caption = f"{block_metadata}"
    text_file_content.append(f"synthetic_image_{img_index + 1}.png${caption}")

    # Save image
    img_path = f"{out_dir}synthetic_image_{img_index + 1}.png"
    img.save(img_path)

# Write text file content
text_file_path = f"{out_dir}captions.txt"
with open(text_file_path, 'w') as f:
    for line in text_file_content:
        f.write(line + "\n")

