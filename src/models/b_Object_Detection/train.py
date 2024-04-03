import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from model import CNNtoRNN
from get_loader import get_data_loaders, Vocabulary  
from tqdm import tqdm
from PIL import Image

def train():
    transform = transforms.Compose(
        [
            transforms.Resize((756, 660)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, _, train_dataset, _ = get_data_loaders(
        train_annotations_file="../../../data/train_test_data/train_imgloc_caption.jsonl",
        test_annotations_file="../../../data/train_test_data/test_imgloc_caption.jsonl",
        root_folder="../../../data",
        transform=transform
    )
    

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    # Hyperparam

    embed_size = 256
    hidden_size = 256
    vocab_size = len(train_dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    writer = SummaryWriter("results/CoordDataset")
    step = 0

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:

        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        #print_examples(model, device, dataset)
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimize": optimizer.state_dict(),
                "step": step,

            }
            save_checkpoint(checkpoint, "results/checkpoint.pth")

        for idx, (imgs, captions) in loop:
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1]) 
            # ()
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())





if __name__ == "__main__":
    train()





