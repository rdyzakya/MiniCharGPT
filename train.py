from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tokenizer import CharTokenizer
from data import CharDS
from model import MiniCharGPTLM

def init_args():
    parser = ArgumentParser()
    # data
    parser.add_argument("--data", type=str, help="Data file path", required=True)
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

def train(model, device, dataloader, epoch, lr):
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    bar = tqdm(total=epoch*len(dataloader), desc="Training")

    for e in range(epoch):
        model.train()
        train_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            inputs, labels = torch.vstack(inputs).T.to(device), labels.to(device)

            out = model.forward(inputs)

            loss = criterion(out, labels)
            train_loss += loss.item() * out.shape[0]

            loss.backward()
            optimizer.step()
            bar.update()
        train_loss /= len(dataloader.dataset)
        print(f"Epoch {e+1} | Train Loss {train_loss}")
    
    return model

def main():
    args = init_args()

    # prepare dataset
    print(f"Prepare dataset from {args.data}...")
    tokenizer = CharTokenizer()
    ds = CharDS.load_data(args.data,
                          tokenizer,
                          dict(truncate=True, padding=True, max_length=args.seq_len))
    dataloader = DataLoader(ds, batch_size=args.batch, shuffle=True)

    # prepare model
    print("Prepareing model...")
    model = MiniCharGPTLM(seqlen=args.seq_len, h_dim=args.d_model, ff_dim=args.ff_dim,
                          n_head=args.n_head, n_block=args.n_block,
                          n_token=len(tokenizer.char2id))
    device = torch.device(f"cuda:{args.gpu}") if (torch.cuda.is_available() and args.gpu != -1) else torch.device("cpu")

    # train
    print("Start training...")
    model = train(model, device, dataloader, args.epoch, args.lr)

    print("Done training, saving model...")

    ckpt = {
        "seqlen" : args.seq_len,
        "h_dim" : args.d_model,
        "ff_dim" : args.ff_dim,
        "n_head" : args.n_head,
        "n_block" : args.n_block,
        "state_dict" : model.state_dict()
    }

    torch.save(ckpt, args.ckpt)

    print(f"Done saving! Can be found at {args.ckpt}")

if __name__ == "__main__":
    main()