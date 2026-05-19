from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tokenizer import CharTokenizer
from data import CharDS
from model import GPT
import time
import json
import os

def init_args():
    parser = ArgumentParser()
    # data
    parser.add_argument("--train_data", type=str, help="Train data file path", required=True)
    # model
    parser.add_argument("--max_length", type=int, help="Max sequence length", default=64)
    parser.add_argument("--dim_model", type=int, help="Model's hidden dimension", default=512)
    parser.add_argument("--n_head", type=int, help="Number of head in multi-head attention", default=8)
    parser.add_argument("--dim_ff", type=int, help="Model's positional ffnn inner dimension", default=2048)
    parser.add_argument("--n_block", type=int, help="Number of decoder blocks", default=4)
    # train
    parser.add_argument("--batch", type=int, help="Training batch size", default=16)
    parser.add_argument("--lr", type=float, help="Adam optimizer's learning rate", default=3e-4)
    parser.add_argument("--epoch", type=int, help="Number of epoch", default=10)
    # save
    parser.add_argument("--ckpt", type=str, help="Model checkpoint's file path", default="model.pth")
    args = parser.parse_args()
    return args

def train(model, device, train_dataloader, epoch, lr):
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
            optimizer.zero_grad()

            for k, v in batch.items():
                batch[k] = v.to(device)

            out = model.forward(**batch) # shape: (num_batch, seq_len, n_vocab)

            # Next Token Prediction
            logits = out[:, :-1, :]        # (B, T-1, V)
            labels = batch["labels"][:, 1:]  # (B, T-1)

            loss = criterion(logits.view(-1, out.shape[-1]), labels.view(-1))
            train_loss += loss.item() * out.shape[0]

            loss.backward()
            optimizer.step()
            train_bar.update()
        train_loss /= len(train_dataloader.dataset)
        train_end_time = time.time()

        train_time = train_end_time - train_start_time
        print(f"Epoch {e+1} | Train Loss {train_loss} | Train Time {train_time:.2f}")

        history.append({
            "epoch" : e + 1,
            "train_loss" : train_loss,
            "train_time" : train_time,
        })
    
    return model, history

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    args = init_args()

    # prepare dataset
    print(f"Prepare dataset from {args.train_data}...")
    train_ds = CharDS(args.train_data)

    # tokenize dataset
    tokenizer = CharTokenizer()
    train_ds.tokenize(tokenizer)

    collate_fn = lambda x: tokenizer.collate(x, truncate=True, padding='longest', max_length=args.max_length)

    train_dataloader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    # prepare model
    print("Preparing model...")
    model = GPT(dim_model=args.dim_model, n_head=args.n_head, dim_ff=args.dim_ff,
                n_block=args.n_block, n_vocab=len(tokenizer.char2id))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train
    print("Start training...")
    model, history = train(model, device, train_dataloader, args.epoch, args.lr)

    print("Done training, saving model...")

    model = model.cpu()

    ckpt = {
        "dim_model" : args.dim_model,
        "dim_ff" : args.dim_ff,
        "n_head" : args.n_head,
        "n_block" : args.n_block,
        "state_dict" : model.state_dict()
    }

    ckpt_dir, _ = os.path.split(args.ckpt)

    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(ckpt, args.ckpt)

    with open(os.path.join(ckpt_dir, "history.json"), 'w') as fp:
        json.dump(history, fp)

    print(f"Done saving! Can be found at {ckpt_dir}")

if __name__ == "__main__":
    main()