import json
import pandas as pd
# df to use for training/testing
def create_df(read_file_path, write_file_path):

    rows = []
    with open(read_file_path, 'r') as file:
        for line in file: 
            item = json.loads(line) 
            image_path = item["image_path"]
            id_ = int(image_path.split('synthetic_image_')[-1].split('.png')[0])
            captions = json.loads(item["caption"].replace("'", "\""))
            for caption in captions:
                row = {
                    "id": id_,
                    "names": caption["name"],
                    "x": caption["keypoint"]["x"],
                    "y": caption["keypoint"]["y"],
                    "img_path": image_path
                }
                rows.append(row)

    df = pd.DataFrame(rows)


    classes = sorted(df['names'].unique())
    cls2id = {cls_name: i for i, cls_name in enumerate(classes)}
    id2cls = {i: cls_name for i, cls_name in enumerate(classes)}

    df['label'] = df['names'].map(cls2id)

    df.to_csv(write_file_path, index=False)  


if __name__ == "__main__":

    train_file_path = 'data/train_imgloc_caption.jsonl'  
    test_file_path = 'data/test_imgloc_caption.jsonl' 

    train_csv_file_path = 'data/train_imgloc_caption.csv'  
    test_csv_file_path = 'data/test_imgloc_caption.csv'

    create_df(train_file_path, train_csv_file_path)
    create_df(test_file_path, test_csv_file_path)