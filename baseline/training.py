import os.path
import argparse
from math import sqrt, exp
import torch as th
from data import MTDataset, MTDataLoader, Vocab
from transformer import Transformer
from tqdm import tqdm


def load_data(src_lang, tgt_lang, cached_folder="assignment2/data", overwrite=False):
    """Load data (and cache to file)"""
    cached_file = os.path.join(cached_folder, f"{src_lang}-{tgt_lang}.pt")
    if not os.path.isfile(cached_file) or overwrite:
        base_folder = os.path.join(
            "assignment2",
            "data",
            f"{src_lang}_{tgt_lang}"
        )
        train_prefix = os.path.join(
            base_folder,
            f"{src_lang}{tgt_lang}_parallel.bpe.train"
        )
        dev_prefix = os.path.join(
            base_folder,
            f"{src_lang}{tgt_lang}_parallel.bpe.dev"
        )
        vocab = Vocab.from_data_files(
            f"{train_prefix}.{src_lang}",
            f"{train_prefix}.{tgt_lang}",
        )
        train = MTDataset(vocab, train_prefix,
                          src_lang=src_lang, tgt_lang=tgt_lang)
        valid = MTDataset(vocab, dev_prefix,
                          src_lang=src_lang, tgt_lang=tgt_lang)
        th.save([vocab, train, valid], cached_file)
    # Load cached dataset
    return th.load(cached_file)


def get_args():
    parser = argparse.ArgumentParser("Train an MT model")
    # General params
    parser.add_argument("--seed", type=int, default=11731)
    parser.add_argument("--src", type=str, default="en", choices=["en"])
    parser.add_argument("--tgt", type=str, default="af",
                        choices=["af", "ts", "nso"])
    parser.add_argument("--model-file", type=str, default="model.pt")
    parser.add_argument("--overwrite-model", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    # Model parameters
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--word-dropout", type=float, default=0.1)
    # Optimization parameters
    parser.add_argument("--n-epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=4e-2)
    parser.add_argument("--lr-decay", type=float, default=0.8)
    parser.add_argument("--inverse-sqrt-schedule", action="store_true")
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--tokens-per-batch", type=int, default=8000)
    parser.add_argument("--samples-per-batch", type=int, default=128)
    return parser.parse_args()


def move_to_device(tensors, device):
    return [tensor.to(device) for tensor in tensors]


def inverse_sqrt_schedule(warmup, lr0):
    """Inverse sqrt learning rate schedule with warmup"""
    step = 0
    # Trick for allowing warmup of 0
    warmup = max(warmup, 0.01)
    while True:
        scale = min(1/sqrt(step+1e-20), step/sqrt(warmup**3))
        step += 1
        yield lr0 * scale


def train_epoch(model, optim, dataloader, lr_schedule=None, clip_grad=5.0):
    # Model device
    device = list(model.parameters())[0].device
    # Iterate over batches
    itr = tqdm(dataloader)
    for batch in itr:
        optim.zero_grad()
        itr.total = len(dataloader)
        # Cast input to device
        batch = move_to_device(batch, device)
        # Various inputs
        src_tokens, src_mask, tgt_tokens, tgt_mask = batch
        # Get log probs
        log_p = model(src_tokens, tgt_tokens[:-1], src_mask)
        # Negative log likelihood of the target tokens
        # (this selects log_p[i, b, tgt_tokens[i+1, b]]
        # for each batch b, position i)
        nll = th.nn.functional.nll_loss(
            # Log probabilities (flattened to (l*b) x V)
            log_p.view(-1, log_p.size(-1)),
            # Target tokens (we start from the 1st real token, ignoring <sos>)
            tgt_tokens[1:].view(-1),
            # Don't compute the nll of padding tokens
            ignore_index=model.vocab["<pad>"],
            # Take the average
            reduction="mean",
        )
        # Perplexity (for logging)
        ppl = th.exp(nll).item()
        # Backprop
        nll.backward()
        # Adjust learning rate with schedule
        if lr_schedule is not None:
            learning_rate = next(lr_schedule)
            for param_group in optim.param_groups:
                param_group["lr"] = learning_rate
        # Gradient clipping
        if clip_grad > 0:
            th.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        # Optimizer step
        optim.step()
        # Update stats
        itr.set_postfix(loss=f"{nll.item():.3f}", ppl=f"{ppl:.2f}")


def evaluate_ppl(model, dataloader):
    model.eval()
    # Model device
    device = list(model.parameters())[0].device
    # total tokens
    tot_tokens = tot_nll = 0
    # Iterate over batches
    for batch in tqdm(dataloader):
        # Cast input to device
        batch = move_to_device(batch, device)
        # Various inputs
        src_tokens, src_mask, tgt_tokens, tgt_mask = batch
        with th.no_grad():
            # Get log probs
            log_p = model(src_tokens, tgt_tokens[:-1], src_mask)
            # Negative log likelihood of the target tokens
            # (this selects log_p[i, b, tgt_tokens[i+1, b]]
            # for each batch b, position i)
            nll = th.nn.functional.nll_loss(
                # Log probabilities (flattened to (l*b) x V)
                log_p.view(-1, log_p.size(-1)),
                # Target tokens (we start from the 1st real token)
                tgt_tokens[1:].view(-1),
                # Don't compute the nll of padding tokens
                ignore_index=model.vocab["<pad>"],
                # Take the average
                reduction="sum",
            )
            # Number of tokens (ignoring <sos> and <pad>)
            n_sos = tgt_tokens.eq(model.vocab["<sos>"]).float().sum().item()
            n_pad = tgt_tokens.eq(model.vocab["<pad>"]).float().sum().item()
            n_tokens = tgt_tokens.numel() - n_pad - n_sos
            # Keep track
            tot_nll += nll.item()
            tot_tokens += n_tokens
    return exp(tot_nll / tot_tokens)


def main():
    # Command line arguments
    args = get_args()
    # Set random seed
    th.manual_seed(args.seed)
    # data
    vocab, train_data, valid_data = load_data(args.src, args.tgt)
    # Model
    model = Transformer(
        args.n_layers,
        args.embed_dim,
        args.hidden_dim,
        args.n_heads,
        vocab,
        args.dropout,
        args.word_dropout,
    )
    if args.cuda:
        model = model.cuda()
    # Load existing model
    if os.path.isfile(args.model_file) and not args.overwrite_model:
        model.load_state_dict(th.load(args.model_file))
    # Optimizer
    optim = th.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    # Learning rate schedule
    lr_schedule = None
    if args.inverse_sqrt_schedule:
        inverse_sqrt_schedule(2000, args.lr)
    # Dataloader
    train_loader = MTDataLoader(
        train_data,
        max_bsz=args.samples_per_batch,
        max_tokens=args.tokens_per_batch,
        shuffle=True
    )
    valid_loader = MTDataLoader(
        valid_data,
        max_bsz=args.samples_per_batch,
        max_tokens=args.tokens_per_batch,
        shuffle=False
    )
    # Either validate
    if args.validate_only:
        valid_ppl = evaluate_ppl(model, valid_loader)
        print(f"Validation perplexity: {valid_ppl:.2f}")
    else:
        # Train epochs
        best_ppl = 1e12
        for epoch in range(1, args.n_epochs+1):
            print(f"----- Epoch {epoch} -----", flush=True)
            # Train for one epoch
            model.train()
            train_epoch(model, optim, train_loader,
                        lr_schedule, args.clip_grad)
            # Check dev ppl
            model.eval()
            valid_ppl = evaluate_ppl(model, valid_loader)
            print(f"Validation perplexity: {valid_ppl:.2f}", flush=True)
            # Early stopping maybe
            if valid_ppl < best_ppl:
                best_ppl = valid_ppl
                print(f"Saving new best model (epoch {epoch} ppl {valid_ppl})")
                th.save(model.state_dict(), args.model_file)
            else:
                for param_group in optim.param_groups:
                    param_group["lr"] *= args.lr_decay


if __name__ == "__main__":
    main()
