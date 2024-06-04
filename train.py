from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tokenizer import CharTokenizer
from data import CharDS, LanguageModelingDataCollator
from model import MiniCharGPTLM
import time
import json
import os

def init_args():
    parser = ArgumentParser()
    # data
    parser.add_argument("--train_data", type=str, help="Train data file path", required=True)
    parser.add_argument("--test_data", type=str, help="Test data file path", required=True)
    # model
    parser.add_argument("--seq_len", type=int, help="Max sequence length", default=64)
    parser.add_argument("--d_model", type=int, help="Model's hidden dimension", default=768)
    parser.add_argument("--ff_dim", type=int, help="Model's positional ffnn inner dimension", default=1024)
    parser.add_argument("--n_head", type=int, help="Number of head in multi-head attention", default=4)
    parser.add_argument("--n_block", type=int, help="Number of decoder blocks", default=4)
    # train
    parser.add_argument("--gpu", type=int, help="GPU ID, -1 for cpu", default=-1)
    parser.add_argument("--batch", type=int, help="Training batch size", default=16)
    parser.add_argument("--lr", type=float, help="SGD optimizer's learning rate", default=3e-4)
    parser.add_argument("--epoch", type=int, help="Number of epoch", default=10)
    # save
    parser.add_argument("--ckpt", type=str, help="Model checkpoint's file path", default="model.pth")
    args = parser.parse_args()
    return args

def train(model, device, train_dataloader, val_dataloader, epoch, lr):
    history = []
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_bar = tqdm(total=epoch*len(train_dataloader), desc="Training")

    for e in range(epoch):
        train_start_time = time.time()
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            labels = batch.pop("labels").to(device)
            input_ids = batch.pop("input_ids").to(device)
            attention_mask = batch.pop("attention_mask").to(device)

            optimizer.zero_grad()

            out = model.forward(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(out.view(-1, out.shape[-1]), labels.view(-1))
            train_loss += loss.item() * out.shape[0]

            loss.backward()
            optimizer.step()
            train_bar.update()
        train_loss /= len(train_dataloader.dataset)
        train_end_time = time.time()

        val_start_time = time.time()
        model.eval()
        val_bar = tqdm(total=len(val_dataloader), desc="Evaluation")
        with torch.no_grad():
            val_loss = 0
            for batch in val_dataloader:
                labels = batch.pop("labels").to(device)
                input_ids = batch.pop("input_ids").to(device)
                attention_mask = batch.pop("attention_mask").to(device)

                out = model.forward(input_ids=input_ids, attention_mask=attention_mask)

                loss = criterion(out.view(-1, out.shape[-1]), labels.view(-1))
                val_loss += loss.item() * out.shape[0]

                val_bar.update()
            val_loss /= len(val_dataloader.dataset)
        val_end_time = time.time()

        train_time = train_end_time - train_start_time
        val_time = val_end_time - val_start_time
        print(f"Epoch {e+1} | Train Loss {train_loss} | Val Loss {val_loss} | Train Time {train_time:.2f} | Val Time {val_time:.2f}")

        history.append({
            "epoch" : e + 1,
            "train_loss" : train_loss,
            "val_loss" : val_loss,
            "train_time" : train_time,
            "val_time" : val_time,
            "train_data" : len(train_dataloader.dataset),
            "val_data" : len(val_dataloader.dataset),
            "train_step" : len(train_dataloader),
            "val_step" : len(val_dataloader)
        })
    
    return model, history

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    args = init_args()

    # prepare dataset
    print(f"Prepare dataset from {args.train_data} and {args.test_data}...")
    tokenizer = CharTokenizer()
    train_ds = CharDS.load_data(args.train_data,
                          tokenizer,
                          dict(truncate=True, padding=True, max_length=args.seq_len))
    
    test_ds = CharDS.load_data(args.test_data,
                          tokenizer,
                          dict(truncate=True, padding=True, max_length=args.seq_len))
    
    collator = LanguageModelingDataCollator(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collator)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch, shuffle=True, collate_fn=collator)

    # prepare model
    print("Preparing model...")
    model = MiniCharGPTLM(h_dim=args.d_model, ff_dim=args.ff_dim,
                          n_head=args.n_head, n_block=args.n_block,
                          n_token=len(tokenizer.char2id))
    device = torch.device(f"cuda:{args.gpu}") if (torch.cuda.is_available() and args.gpu != -1) else torch.device("cpu")

    # train
    print("Start training...")
    model, history = train(model, device, train_dataloader, test_dataloader, args.epoch, args.lr)

    print("Done training, saving model...")

    model = model.cpu()

    ckpt = {
        "h_dim" : args.d_model,
        "ff_dim" : args.ff_dim,
        "n_head" : args.n_head,
        "n_block" : args.n_block,
        "state_dict" : model.state_dict()
    }

    ckpt_dir, _ = os.path.split(args.ckpt)

    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(ckpt, args.ckpt)

    stats = {
        "h_dim" : args.d_model,
        "ff_dim" : args.ff_dim,
        "n_head" : args.n_head,
        "n_block" : args.n_block,
        "params" : count_parameters(model),
        "history" : history
    }

    with open(os.path.join(ckpt_dir, "stats.json"), 'w') as fp:
        json.dump(stats, fp)

    print(f"Done saving! Can be found at {ckpt_dir}")

if __name__ == "__main__":
    main()