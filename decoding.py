"""
decoding.py  -  inference tools for the EN->AR Transformer
  * beam_search_decode : batched beam search (length penalty + n-gram repeat block)
  * Translator         : loads model + tokenizers from Drive, translate(text)
  * show_examples      : greedy vs beam side by side on validation sentences
  * evaluate_bleu      : sacreBLEU for greedy vs beam
Put this file next to config.py / model.py / dataset.py / train.py.
"""
import re
import torch


from tokenizers import Tokenizer
from config import get_config, get_latest_weights_path
from dataset import causal_mask
from train import get_model, greedy_decode


# ------------------------------------------------------------------ helpers
def detok(text):
    """Undo the spaces the Whitespace pre-tokenizer puts around punctuation."""
    text = re.sub(r"\s+([،,.؟?!:;)\]»])", r"\1", text)
    text = re.sub(r"([(\[«])\s+", r"\1", text)
    return text.strip()


def _banned_tokens(seq, n):
    """Tokens that would create an n-gram already present in seq."""
    if n <= 0 or len(seq) < n:
        return set()
    prefix = tuple(seq[len(seq) - (n - 1):]) if n > 1 else ()
    return {
        seq[i + n - 1]
        for i in range(len(seq) - n + 1)
        if tuple(seq[i:i + n - 1]) == prefix
    }


# ------------------------------------------------------------------ beam search
@torch.no_grad()
def beam_search_decode(model, source, source_mask, tokenizer_tgt, max_len, device,
                       beam_size=4, length_penalty=0.6, no_repeat_ngram=3):
    """
    source: (1, seq_len)   source_mask: (1, 1, 1, seq_len)
    Returns a 1-D tensor of token ids: [SOS] ... [EOS]
    Scores are summed log-probs (the model's projection layer returns log_softmax).
    """
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    def lp(length):  # GNMT length penalty
        return ((5.0 + length) / 6.0) ** length_penalty

    encoder_output = model.encode(source, source_mask)            # (1, S, D) - computed once

    beams = torch.full((1, 1), sos_idx, dtype=torch.long, device=device)   # (k, t)
    scores = torch.zeros(1, device=device)                                 # (k,)
    finished = []                                                          # (normalized_score, tokens)

    while beams.size(1) < max_len:
        k = beams.size(0)
        enc = encoder_output.expand(k, -1, -1)
        smask = source_mask.expand(k, -1, -1, -1)
        tmask = causal_mask(beams.size(1)).to(device)

        out = model.decode(beams, enc, smask, tmask)
        logp = model.project(out[:, -1]).float()                  # (k, V)

        if beams.size(1) == 1:                                    # no empty translations
            logp[:, eos_idx] = float("-inf")
        if no_repeat_ngram > 0 and beams.size(1) >= no_repeat_ngram:
            for i in range(k):
                banned = _banned_tokens(beams[i].tolist(), no_repeat_ngram)
                if banned:
                    logp[i, list(banned)] = float("-inf")

        cand = scores.unsqueeze(1) + logp                         # (k, V)
        V = cand.size(1)
        top_scores, top_idx = cand.view(-1).topk(min(2 * beam_size, cand.numel()))

        new_beams, new_scores = [], []
        for s, idx in zip(top_scores.tolist(), top_idx.tolist()):
            if s == float("-inf"):
                continue
            b, tok = divmod(idx, V)
            seq = torch.cat([beams[b], torch.tensor([tok], device=device)])
            if tok == eos_idx:
                finished.append((s / lp(seq.size(0)), seq))
            else:
                new_beams.append(seq)
                new_scores.append(s)
                if len(new_beams) == beam_size:
                    break

        if not new_beams or len(finished) >= beam_size:
            break
        beams = torch.stack(new_beams)
        scores = torch.tensor(new_scores, device=device)

    if not finished:                                              # hit max_len without EOS
        finished = [(scores[0].item() / lp(beams.size(1)), beams[0])]
    return max(finished, key=lambda x: x[0])[1]


# ------------------------------------------------------------------ model loading
def load_model_and_tokenizers(config, device, weights_path=None):
    tok_src = Tokenizer.from_file(config["tokenizer_file"].format("en"))
    tok_tgt = Tokenizer.from_file(config["tokenizer_file"].format("ar"))
    model = get_model(config, tok_src.get_vocab_size(), tok_tgt.get_vocab_size()).to(device)

    path = weights_path or get_latest_weights_path(config)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"Loaded {path}  (trained epochs: {state['epoch'] + 1})")
    return model, tok_src, tok_tgt


class Translator:
    def __init__(self, config=None, device=None, weights_path=None):
        self.config = config or get_config()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tok_src, self.tok_tgt = load_model_and_tokenizers(
            self.config, self.device, weights_path
        )
        self.seq_len = self.config["seq_len"]
        # BilingualDataset builds BOTH encoder and decoder inputs with the target tokenizer's
        # special ids, so we do exactly the same here.
        self.sos = self.tok_tgt.token_to_id("[SOS]")
        self.eos = self.tok_tgt.token_to_id("[EOS]")
        self.pad = self.tok_tgt.token_to_id("[PAD]")

    # text -> (src, mask), same layout as BilingualDataset
    def encode_source(self, text):
        if len(text.split()) > self.config.get("max_words", 40):
            print(f"warning: input is longer than the {self.config.get('max_words', 40)} words "
                  "the model was trained on, quality will drop")
        ids = self.tok_src.encode(text.strip()).ids[: self.seq_len - 2]
        n_pad = self.seq_len - len(ids) - 2
        src = torch.tensor([self.sos] + ids + [self.eos] + [self.pad] * n_pad, dtype=torch.int64)
        src = src.unsqueeze(0).to(self.device)                          # (1, seq_len)
        mask = (src != self.pad).unsqueeze(0).unsqueeze(0).int()        # (1, 1, 1, seq_len)
        return src, mask

    @torch.no_grad()
    def decode(self, src, mask, method="beam", **beam_kwargs):
        self.model.eval()
        if method == "greedy":
            out = greedy_decode(self.model, src, mask, self.tok_src, self.tok_tgt,
                                self.seq_len, self.device)
        else:
            out = beam_search_decode(self.model, src, mask, self.tok_tgt,
                                     self.seq_len, self.device, **beam_kwargs)
        text = self.tok_tgt.decode(out.tolist(), skip_special_tokens=True)
        return detok(text)

    def translate(self, text, method="beam", **beam_kwargs):
        src, mask = self.encode_source(text)
        return self.decode(src, mask, method, **beam_kwargs)


_default = None


def translate(text, method="beam", **kw):
    """Convenience: translate("Hello") - loads the latest checkpoint on first call."""
    global _default
    if _default is None:
        _default = Translator()
    return _default.translate(text, method, **kw)


# ------------------------------------------------------------------ evaluation
def show_examples(translator, val_dl, n=5, beam_size=4):
    """val_dl must have batch_size=1 (the one returned by train.load_data)."""
    for i, batch in enumerate(val_dl):
        if i == n:
            break
        src = batch["encoder_input"].to(translator.device)
        mask = batch["encoder_attention_mask"].to(translator.device)
        print("-" * 80)
        print(f"{'SOURCE:':>20} {batch['src_text'][0]}")
        print(f"{'TARGET:':>20} {batch['tgt_text'][0]}")
        print(f"{'GREEDY:':>20} {translator.decode(src, mask, 'greedy')}")
        print(f"{'BEAM:':>20} {translator.decode(src, mask, 'beam', beam_size=beam_size)}")
    print("-" * 80)


def evaluate_bleu(translator, val_dl, n=500, beam_size=4, **beam_kwargs):
    import sacrebleu  # pip install sacrebleu
    refs, greedy_h, beam_h = [], [], []
    for i, batch in enumerate(val_dl):
        if i == n:
            break
        src = batch["encoder_input"].to(translator.device)
        mask = batch["encoder_attention_mask"].to(translator.device)
        refs.append(batch["tgt_text"][0])
        greedy_h.append(translator.decode(src, mask, "greedy"))
        beam_h.append(translator.decode(src, mask, "beam", beam_size=beam_size, **beam_kwargs))

    g = sacrebleu.corpus_bleu(greedy_h, [refs])
    b = sacrebleu.corpus_bleu(beam_h, [refs])
    print(f"sentences: {len(refs)}")
    print(f"greedy BLEU: {g.score:.2f}")
    print(f"beam   BLEU: {b.score:.2f}   (beam_size={beam_size}, {beam_kwargs})")
    return g, b