from pathlib import Path
import os, re, warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import BilingualDataset, causal_mask
from model import Transformer
from config import get_config, get_weights_file_path, get_latest_weights_path

warnings.filterwarnings("ignore", category=UserWarning, module="datasets")


# ---------------------------------------------------------------- decoding
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_seq_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.full((1, 1), sos_idx, dtype=source.dtype, device=device)

    while decoder_input.size(1) < max_seq_len:
        tgt_mask = causal_mask(decoder_input.size(1)).to(device)
        out = model.decode(decoder_input, encoder_output, source_mask, tgt_mask)
        prob = model.project(out[:, -1, :])
        next_token = torch.argmax(prob, dim=-1)
        decoder_input = torch.cat(
            [decoder_input, torch.full((1, 1), next_token.item(), dtype=source.dtype, device=device)], dim=1
        )
        if next_token.item() == eos_idx:
            break
    return decoder_input.squeeze(0)


# ---------------------------------------------------------------- validation
def run_validation(model, val_dl, tokenizer_src, tokenizer_tgt, max_seq_len, device, print_msg, num_examples=3):
    model.eval()
    count = 0
    with torch.no_grad():
        for batch in val_dl:
            count += 1
            enc_in = batch["encoder_input"].to(device)
            enc_mask = batch["encoder_attention_mask"].to(device)
            assert enc_in.size(0) == 1, "Batch size must be 1 for greedy decoding."

            out = greedy_decode(model, enc_in, enc_mask, tokenizer_src, tokenizer_tgt, max_seq_len, device)
            pred = tokenizer_tgt.decode(out.tolist(), skip_special_tokens=True)

            print_msg(f"\nExample {count}:")
            print_msg(f"Source:    {batch['src_text'][0]}")
            print_msg(f"Target:    {batch['tgt_text'][0]}")
            print_msg(f"Predicted: {pred}")
            if count == num_examples:
                break


def validation_loss(model, val_dl, loss_fn, device, max_batches=100):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, b in enumerate(val_dl):
            if i == max_batches:
                break
            enc_in, dec_in = b["encoder_input"].to(device), b["decoder_input"].to(device)
            em, dm = b["encoder_attention_mask"].to(device), b["decoder_attention_mask"].to(device)
            out = model.project(model.decode(dec_in, model.encode(enc_in, em), em, dm))
            total += loss_fn(out.view(-1, out.size(-1)), b["labels"].to(device).view(-1)).item()
            n += 1
    return total / max(n, 1)


# ---------------------------------------------------------------- data
def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang] if "translation" in item else item[lang]


def get_or_build_tokenizer(config, ds, lang):
    path = Path(config["tokenizer_file"].format(lang))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tokenizer = Tokenizer(BPE(unk_token="[UNK]", end_of_word_suffix="</w>"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")
        trainer = BpeTrainer(
            vocab_size=config["vocab_size"],
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[EOS]", "[SOS]", "[BOS]", "[MASK]"],
            end_of_word_suffix="</w>",
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(path))
    else:
        tokenizer = Tokenizer.from_file(str(path))
    return tokenizer


AR_NOISE = re.compile(r"[\u0640\u064B-\u0652]")  # tatweel + diacritics


def clean_pair(x):
    x["translation"]["ar"] = AR_NOISE.sub("", x["translation"]["ar"]).strip()
    x["translation"]["en"] = x["translation"]["en"].strip()
    return x


def load_data(config):
    ds_raw = load_dataset("Helsinki-NLP/opus-100", "ar-en", split="train")
    ds_raw = ds_raw.shuffle(seed=config["seed"]).select(range(config["num_samples"]))
    ds_raw = ds_raw.map(clean_pair)

    mw = config["max_words"]
    ds_raw = ds_raw.filter(
        lambda x: 0 < len(x["translation"]["en"].split()) <= mw
        and 0 < len(x["translation"]["ar"].split()) <= mw
    )

    n_train = int(0.9 * len(ds_raw))
    gen = torch.Generator().manual_seed(config["seed"])  # same split every run
    train_raw, val_raw = random_split(ds_raw, [n_train, len(ds_raw) - n_train], generator=gen)

    src_tok = get_or_build_tokenizer(config, train_raw, "en")
    tgt_tok = get_or_build_tokenizer(config, train_raw, "ar")

    # drop pairs that don't fit in seq_len TOKENS (encoder needs +2 for SOS/EOS, decoder +1)
    trans = ds_raw["translation"]
    limit = config["seq_len"] - 2

    def keep_short(subset):
        idx = subset.indices
        en = src_tok.encode_batch([trans[i]["en"] for i in idx])
        ar = tgt_tok.encode_batch([trans[i]["ar"] for i in idx])
        keep = [i for i, e, a in zip(idx, en, ar) if len(e.ids) <= limit and len(a.ids) <= limit]
        print(f"Kept {len(keep)}/{len(idx)} pairs after token-length filter")
        return Subset(ds_raw, keep)

    train_raw = keep_short(train_raw)
    val_raw = keep_short(val_raw)

    train_ds = BilingualDataset(train_raw, src_tok, tgt_tok, "en", "ar", config["seq_len"])
    val_ds = BilingualDataset(val_raw, src_tok, tgt_tok, "en", "ar", config["seq_len"])

    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)          # greedy decode needs batch=1
    val_loss_dl = DataLoader(val_ds, batch_size=64, shuffle=False)    # for validation loss
    return train_dl, val_dl, val_loss_dl, src_tok, tgt_tok

# ---------------------------------------------------------------- model / checkpoints
def get_model(config, vs, vt):
    return Transformer.build_transformer(
        src_vocab_size=vs, tgt_vocab_size=vt,
        src_seq_len=config["seq_len"], tgt_seq_len=config["seq_len"],
        d_model=config["d_model"], N=config["num_encoder_layers"],
        h=config["num_heads"], dropout=config["dropout"], d_ff=config["dim_feedforward"],
    )


def save_checkpoint(config, model, optimizer, scheduler, scaler, epoch, global_step):
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "global_step": global_step,
    }
    # write to temp then rename -> a disconnect mid-save can't corrupt the file
    for path in (get_weights_file_path(config, epoch), get_latest_weights_path(config)):
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)

    # prune old epoch files to save Drive space
    old = epoch - config["keep_last"]
    if old >= 0:
        p = Path(get_weights_file_path(config, old))
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------- training
def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, val_loss_dl, src_tok, tgt_tok = load_data(config)
    model = get_model(config, src_tok.get_vocab_size(), tgt_tok.get_vocab_size()).to(device)
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
    w = config["warmup_steps"]
    scheduler = LambdaLR(optimizer, lambda s: min((s + 1) / w, (w / (s + 1)) ** 0.5))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # auto-resume from latest checkpoint on Drive
    initial_epoch, global_step = 0, 0
    latest = get_latest_weights_path(config)
    if Path(latest).exists():
        print(f"Resuming from {latest}")
        state = torch.load(latest, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_tok.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        it = tqdm(train_dl, desc=f"Epoch {epoch+1}/{config['num_epochs']}", unit="batch")
        running, n = 0.0, 0

        for batch in it:
            enc_in = batch["encoder_input"].to(device)
            dec_in = batch["decoder_input"].to(device)
            enc_mask = batch["encoder_attention_mask"].to(device)
            dec_mask = batch["decoder_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                enc_out = model.encode(enc_in, enc_mask)
                dec_out = model.decode(dec_in, enc_out, enc_mask, dec_mask)  # src_mask first, then tgt_mask
                proj = model.project(dec_out)
            loss = loss_fn(proj.float().view(-1, proj.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            running += loss.item()
            n += 1
            it.set_postfix({"loss": f"{loss.item():.3f}", "lr": f"{scheduler.get_last_lr()[0]:.1e}"})
            if global_step % 50 == 0:
                writer.add_scalar("Loss/train", loss.item(), global_step)

        val_loss = validation_loss(model, val_loss_dl, loss_fn, device)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.flush()
        print(f"Epoch {epoch+1}: train {running/n:.3f} | val {val_loss:.3f}")

        run_validation(model, val_dl, src_tok, tgt_tok, config["seq_len"], device, it.write)
        save_checkpoint(config, model, optimizer, scheduler, scaler, epoch, global_step)
        print(f"Saved checkpoint for epoch {epoch} to Drive")


if __name__ == "__main__":
    train_model(get_config())