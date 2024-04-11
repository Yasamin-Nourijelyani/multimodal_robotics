from src.models.c_robot_perception.model import pix2seq, llm, plot_keypoint

def main(instruction, img_path, keypoint_img_path):
    """
    Running the model that inputs an image and corresponding  
    
    <instruction>: the text description of the instruction
    <img_path>: the path to the image corresponding to the instructin
    <keypoint_img_path>: Location for where to save the image with the keypoint as predicted by the model 
    on the image coresponding to the instruction 
    
    """
    
    # path for vocab 
    test_csv_file_path = 'src/models/b_Object_Detection/pix2seq/data/test_imgloc_caption.csv'
    # running the model pix2seq > llm
    text = pix2seq(img_path, test_csv_file_path)
    extracted_dict = llm(text, instruction)
    print(extracted_dict)
    #final results plotted
    plot_keypoint(img_path, extracted_dict, keypoint_img_path)


if __name__ == "__main__":

    # to be changed for inference:
    instruction = "Locate the green blocks that is near the back, just under two blue blocks."
    # locate an image from the test dataset: 'models/b_Object_Detection/pix2seq/data/test_imgloc_caption.jsonl' 
    img_path = """data/coord_text_images_non_random/images/synthetic_image_3.png"""
    # where to save image after plotting keypoint
    keypoint_img_path = """data/modified_images/synthetic_image_3.png"""

    main(instruction, img_path, keypoint_img_path)