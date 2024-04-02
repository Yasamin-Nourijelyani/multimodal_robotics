# Data

This is a custom dataset used to train a multimodal Vision and Language Model for improving robotic perception.

## Random Block Location Images
To create images with blocks in random locations on the table, run:
```python3 random_block_images.py```
This data is used to train the image captioning model which gives the keypoint location of objects in the image through the caption.

## Non-Random Block Location Images
To create images with blocks in x and locations from block_location.csv on the table, run:
```python3 non_random_block_images.py```
This data used as an intermediate step to the data used to train the LLM. 

## Text Corpus
This is the instruction as well as the coordinate description of the correponding image. 
To get this data run:
```python3 text_only_data_generation.py```
block_location.csv is is originally from: https://github.com/personalrobotics/collaborative_manipulation_corpus
This data is used to train the LLM.