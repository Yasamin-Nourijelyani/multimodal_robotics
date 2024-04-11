# Robotics Visual Perception Enhancement

The goal of this project is to enhance perception capabilities of robots by fine tuning a multimodal vision-language model.
Note: inside each directory, there is a readme discussing the files and purpose of the files in that directory. 

Please see the video of our project here:
[Insert video here]

# Model Architecture:

[Insert image here]

# Basic setup before running the code

The following setup is required before running the code.

```
git clone https://github.com/Yasamin-Nourijelyani/multimodal_robotics.git

cd multimodal_robotics

pip install -r requirements.txt
```


## Running Code on Your Examples

Note, please connect to a 15G or higher RAM GPU (even for inference).

Also note the model might be slow (can take up to a minute for inference).

Note: the model assumes input images are 384 by 384 px. 

In ```src/model.py``` update the 

```instruction```: the text description of the instruction

```img_path```: the path to the image corresponding to the instruction

```keypoint_img_path```: Location for where to save the image with the keypoint as predicted by the model on the image coresponding to the instruction.   

### Run
```
python3 -m src.run
```





