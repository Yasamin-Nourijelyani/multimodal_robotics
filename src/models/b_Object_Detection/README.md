# Object Detection model

This model takes as input images and returns the text coordinates of all the blocks and their colors as captions. 
It is inspired from https://arxiv.org/abs/2109.10852 and https://github.com/moein-shariatnia/Pix2Seq


```create_data.py``` is used to generate data to train and test the pix2seq model: Images and corresponding coordinates of the objects of interest. 

This model tokenizes the text so that we appropriately generate tokens for keypoint locations in the image. It is inspired from image captioning and object detection models.