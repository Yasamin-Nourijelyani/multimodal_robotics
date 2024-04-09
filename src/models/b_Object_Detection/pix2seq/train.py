
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup
from config import CFG
import utils
from tokenizer import KeypointTokenizer, get_loaders
from model import Encoder, Decoder, EncoderDecoder

def train_epoch(model, train_loader, optimizer, lr_scheduler, criterion, logger=None):
    model.train()
    loss_meter = utils.AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    
    for x, y in tqdm_object:
        x, y = x.to(CFG.device, non_blocking=True), y.to(CFG.device, non_blocking=True)
        
        y_input = y[:, :-1]
        y_expected = y[:, 1:]
        

        preds = model(x, y_input)
        loss = criterion(preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        loss_meter.update(loss.item(), x.size(0))
        
        lr = utils.get_lr(optimizer)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=f"{lr:.6f}")
        if logger is not None:
            logger.log({"train_step_loss": loss_meter.avg, 'lr': lr})
    
    return loss_meter.avg


def valid_epoch(model, valid_loader, criterion):
    model.eval()
    loss_meter = utils.AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    
    with torch.no_grad():
        for x, y in tqdm_object:
            x, y = x.to(CFG.device, non_blocking=True), y.to(CFG.device, non_blocking=True)

            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            preds = model(x, y_input)
            loss = criterion(preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))


            loss_meter.update(loss.item(), x.size(0))
    
    return loss_meter.avg


def train_eval(model, 
               train_loader,
               valid_loader,
               criterion, 
               optimizer, 
               lr_scheduler,
               step,
               logger):
    
    best_loss = float('inf')
    
    for epoch in range(CFG.epochs):
        print(f"Epoch {epoch + 1}")
        if logger is not None:
            logger.log({"Epoch": epoch + 1})
        
        train_loss = train_epoch(model, train_loader, optimizer, 
                                 lr_scheduler if step == 'batch' else None, 
                                 criterion, logger=logger)
        
        valid_loss = valid_epoch(model, valid_loader, criterion)
        print(f"Valid loss: {valid_loss:.3f}")
        
        if step == 'epoch':
            pass
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'best_valid_loss.pth')
            print("Saved Best Model")
        
        if logger is not None:
            logger.log({
                'train_loss': train_loss,
                'valid_loss': valid_loss
            })
            logger.save('best_valid_loss.pth')

    
    

if __name__ == "__main__":


    train_csv_file_path = 'data/train_imgloc_caption.csv'  
    test_csv_file_path = 'data/test_imgloc_caption.csv'

    train_df = pd.read_csv(train_csv_file_path)
    valid_df = pd.read_csv(test_csv_file_path)


    classes = sorted(train_df['names'].unique())
    cls2id = {cls_name: i for i, cls_name in enumerate(classes)}
    id2cls = {i: cls_name for i, cls_name in enumerate(classes)}
    num_classes = len(classes)  # num of unique labels
    num_bins = CFG.num_bins     # num of bins for quantization
    width = CFG.img_size       
    height = CFG.img_size       
    max_len = CFG.max_len       # maximum sequence length

    tokenizer = KeypointTokenizer(num_classes=num_classes, num_bins=num_bins,
                                width=width, height=height, max_len=max_len)

    CFG.pad_idx = tokenizer.PAD_code


    train_loader, valid_loader = get_loaders(train_df, valid_df, tokenizer, CFG.img_size, CFG.batch_size, CFG.max_len, tokenizer.PAD_code)


    encoder = Encoder(model_name=CFG.model_name, pretrained=True, out_dim=256)
    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                    encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(encoder, decoder)
    model.to(CFG.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    num_training_steps = CFG.epochs * (len(train_loader.dataset) // CFG.batch_size)
    num_warmup_steps = int(0.05 * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_training_steps=num_training_steps,
                                                num_warmup_steps=num_warmup_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=CFG.pad_idx)

    train_eval(model,
            train_loader,
            valid_loader,
            criterion,
            optimizer,
            lr_scheduler=lr_scheduler,
            step='batch',
            logger=None)