from src.models.c_robot_perception.model import pix2seq, llm, plot_keypoint
import pandas as pd

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
    return extracted_dict


if __name__ == "__main__":

    # # Uncomment for inference on One Image
    # # to be changed for inference:
    # instruction = "Locate the orange block that is near the front, just in front of the yellow and blue block."
    # # locate an image from the test dataset: 'models/b_Object_Detection/pix2seq/data/test_imgloc_caption.jsonl' 
    # img_path = """data/testing/images/synthetic_image_1.png"""
    # # where to save image after plotting keypoint
    # keypoint_img_path = """data/testing_output/output_synthetic_image_1.png"""

    # extracted_dict = main(instruction, img_path, keypoint_img_path)

    answer_df = pd.DataFrame(columns=['Ground Truth x', 'Ground Truth y', 'Predicted x', 'Predicted y', 'abs_dx', 'abs_dy'])
    df = pd.read_csv('data/testing/captions.csv')
    i = 0
    for index, row in df.iterrows():
        img_path = "data/testing/images/" + row['image'] 
        instruction = row['caption']
        keypoint_img_path = "data/testing_output/images/" + row['image'] 
        extracted_dict = main(instruction, img_path, keypoint_img_path)

        # predicted x and y
        predx = extracted_dict["x"]
        predy = extracted_dict["y"]



        answer = row['answer']
        xyreal = answer.split(";")

        # ground truth x and y values
        realx = xyreal[0]
        realy = xyreal[1]

  

        dx = abs(float(realx) - float(predx))
        dy = abs(float(realy) - float(predy))


        answer_df = answer_df.append({'Ground Truth x': realx, 'Ground Truth y': realy, 'Predicted x':predx, 'Predicted y':predy, 'abs_dx': dx, 'abs_dy':dy}, ignore_index=True)
        i += 1
        print(f"saved row {i}")


    answer_df.to_csv("data/testing_output/answer.csv", index=False)



        