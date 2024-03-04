import cv2 as cv
import os
import pandas as pd

block_location = pd.read_csv('./block_location.csv')


# use monospace font and red color
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (0, 0, 255)


# load all png images from './orig_without_arrows' folder

dir = './orig_without_arrows'
dir_1 = './orig_with_arrows'

out_dir = './lab_without_arrows'
out_dir_1 = './lab_with_arrows'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if not os.path.exists(out_dir_1):
    os.makedirs(out_dir_1)

images = [cv.imread(os.path.join(dir, f)) for f in os.listdir(dir) if f.endswith('.png')]

for i, img in enumerate(images):
    
    # img_config is labeled as "Configuration_01.png" and "Configuration_02.png"
    img_config_id = int(os.listdir(dir)[i].split('_')[1].split('.')[0])

    # add labels at block locations
    for j, block in enumerate(block_location.values):
        # check if block_location is for the current image
        if block[0] != img_config_id + 1:
            continue
        # add label at the block location
        # columns are labeled block_id and x, y
        cv.putText(img, str(block[1]), (int(block[2]), int(block[3])), font, font_scale, font_color, 2)

    # save the image in the out_dir
    cv.imwrite(os.path.join(out_dir, f'lab_{img_config_id}.png'), img)

# load all png images from './orig_with_arrows' folder
images = [cv.imread(os.path.join(dir_1, f)) for f in os.listdir(dir_1) if f.endswith('.png')]

for i, img in enumerate(images):
    img_config_id = int(os.listdir(dir_1)[i].split('_')[-2].lstrip('Configuration'))
    img_ver_id = int(os.listdir(dir_1)[i].split('_')[-1].split('.')[0].lstrip('v'))

    # add labels at block locations
    for j, block in enumerate(block_location.values):
        # check if block_location is for the current image
        if block[0] != img_config_id + 1:
            continue
        # add label at the block location
        # columns are labeled block_id and x, y
        cv.putText(img, str(block[1]), (int(block[2]), int(block[3])), font, font_scale, font_color, 2)

    # save the image in the out_dir
    cv.imwrite(os.path.join(out_dir_1, f'lab_{img_config_id}_v{img_ver_id}.png'), img)
