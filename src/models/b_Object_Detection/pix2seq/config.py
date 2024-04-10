import torch

class CFG:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    max_len = 300
    img_size = 384
    num_bins = img_size
    blocks_per_image = 10

    num_classes = 4 # number of colors of boxes
    
    batch_size = 32
    epochs = 100
    
    model_name = 'deit3_small_patch16_384_in21ft1k'
    num_patches = 576
    lr = 1e-4
    weight_decay = 1e-4

    generation_steps = 101

    top_k = 0
    top_p=1

    GT_COLOR = (0, 255, 0) # Green
    PRED_COLOR = (255, 0, 0) # Red
    TEXT_COLOR = (255, 255, 255) # White