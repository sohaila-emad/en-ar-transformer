# English → Arabic Translator (Transformer from scratch)

A sequence-to-sequence **Transformer built from scratch in PyTorch** (encoder–decoder, no pretrained weights) that translates **English to Arabic**. It is trained on a 300k-pair sample of OPUS-100 and comes with beam-search decoding, BLEU evaluation and attention visualizations.

**Model weights:** [Sohaila052/English_to_Arabic_translator on Hugging Face](https://huggingface.co/Sohaila052/English_to_Arabic_translator)

> **Status: educational project.** The model produces fluent Arabic and handles short, common sentences well, but it is not a production translator (BLEU ≈ 6, see [Results](#results) and [Limitations](#limitations)). The goal was to understand every part of the Transformer, from masks to decoding.

---

## Table of contents
1. [Highlights](#highlights)
2. [Model](#model)
3. [Data](#data)
4. [Training](#training)
5. [Results](#results)
6. [Attention analysis](#attention-analysis)
7. [Quick start](#quick-start)
8. [Repository structure](#repository-structure)
9. [What went wrong along the way](#what-went-wrong-along-the-way)
10. [Limitations](#limitations)
11. [Future work](#future-work)
12. [Acknowledgements and references](#acknowledgements-and-references)

---

## Highlights

- Full encoder–decoder Transformer written from scratch (embeddings, sinusoidal positions, multi-head attention, pre-LayerNorm residual blocks, projection layer).
- Custom **BPE tokenizers** (16k vocabulary per language) with Arabic normalization.
- Trained on **~261k sentence pairs** for 12 epochs with warmup + inverse-sqrt LR, fp16 mixed precision and gradient clipping.
- **Resumable training** on Colab: atomic checkpoints on Google Drive that include optimizer, scheduler and scaler state.
- **Beam search** with length penalty and n-gram repeat blocking, improving BLEU from **4.75 → 6.28** over greedy decoding.
- **Attention visualizations** (encoder, decoder and cross attention) that show real source–target alignment.

---

## Model

| Setting | Value |
|---|---|
| Architecture | Transformer encoder–decoder (pre-LayerNorm) |
| `d_model` | 512 |
| Attention heads | 8 |
| Encoder / decoder layers | 5 / 5 |
| Feed-forward size | 2048 |
| Dropout | 0.1 |
| Positional encoding | Sinusoidal |
| Vocabulary | 16k BPE for English, 16k BPE for Arabic |
| Max sequence length | 160 tokens |
| Parameters | ≈ 61 M |

```
English text
   │  BPE tokenizer (en)
   ▼
[SOS] tokens [EOS] [PAD]…  ──►  Encoder (5 layers)  ──┐
                                                       │ cross-attention
[SOS] Arabic tokens so far ──►  Decoder (5 layers)  ◄──┘
                                       │
                                  Linear + log-softmax
                                       │
                          next Arabic token  ──►  BPE detokenizer (ar) ──► Arabic text
```

---

## Data

- **Dataset:** [OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100), `ar-en` configuration (mostly subtitles and UN documents).
- **Sample:** 300,000 random pairs (fixed seed).
- **Cleaning:** removed Arabic tatweel (`ـ`) and diacritics; dropped pairs with an empty side or more than 40 words on either side.
- **Split:** 90% train / 10% validation (seeded, so the split is identical after a Colab disconnect).
- **Token-length filter:** pairs that do not fit in 160 BPE tokens are removed (3 of 261,193 training pairs).
- **Final sizes:** 261,190 training pairs and 29,022 validation pairs.

OPUS-100 is noisy: some pairs are misaligned or very free translations, which limits both training quality and BLEU.

---

## Training

| Setting | Value |
|---|---|
| Optimizer | Adam (β = 0.9, 0.98, ε = 1e-9) |
| Peak learning rate | 3e-4 |
| LR schedule | Linear warmup (4,000 steps), then inverse-square-root decay |
| Batch size | 32 |
| Loss | Cross-entropy, label smoothing 0.1, padding ignored |
| Precision | fp16 mixed precision (AMP) |
| Gradient clipping | 1.0 |
| Epochs | 12 (97,956 steps) |
| Hardware | One Colab GPU, ≈ 35 min per epoch (≈ 7 hours total) |

### Loss curves

![Training and validation loss](assets/loss_curves.png)

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 6.902 | 6.096 |
| 2 | 5.917 | 5.758 |
| 4 | 5.490 | 5.527 |
| 6 | 5.281 | 5.431 |
| 8 | 5.142 | 5.378 |
| 10 | 5.042 | 5.347 |
| 12 | 4.960 | 5.331 |

Validation loss never went up (no overfitting), but improvements were down to about 0.01 per epoch by the end. The loss includes the constant offset added by label smoothing, so the absolute values look higher than the true per-token negative log-likelihood.

---

## Results

### BLEU

Evaluated with [sacreBLEU](https://github.com/mjpost/sacrebleu) (default settings, single reference) on held-out validation sentences, using the final epoch-12 checkpoint.

| Decoding | BLEU |
|---|---:|
| Greedy | 4.75 |
| Beam search (k = 4, 3-gram repeat blocking) | **6.28** |

(500 validation sentences.)

![BLEU results](assets/bleu_results.png)

**Beam-size / repeat-blocking sweep** (300 validation sentences, greedy = 4.99):

| Beam size | No repeat blocking | 3-gram repeat blocking |
|---:|---:|---:|
| 1 | 4.99 | 6.02 |
| 2 | 5.77 | 6.81 |
| 4 | 6.14 | 6.34 |
| 6 | 6.50 | 6.51 |

**Length penalty** (beam 4): 0.0 → 6.30, 0.6 → 6.34, 1.0 → 6.38.

What the sweep shows:

- Beam size 1 without blocking reproduces greedy exactly (4.99), a useful sanity check of the beam-search code.
- Blocking repeated 3-grams alone gives about **+1 BLEU**, because greedy decoding often falls into loops.
- Wider beams add roughly another **+1 BLEU** and already avoid most loops, so blocking matters less at beam 6.
- Length penalty made no measurable difference. On 300 sentences, differences under about 0.5 BLEU are within noise, so the 6.81 at beam 2 should not be read as "beam 2 is best".

> **Caveat:** the validation set is a held-out slice of the training sample, not the official OPUS-100 test split, and the references are single and noisy. Use the numbers to compare decoding methods, not to compare against published systems.

### Example translations

| English | Model (beam) | Quality |
|---|---|---|
| I will see you tomorrow. | سوف أراك غدا | ✅ correct |
| How are you doing today? | كيف حالك اليوم؟ | ✅ correct |
| Do you know what it was? | هل تعرف ماذا كان؟ | ✅ correct |
| Good. | جيد | ✅ correct |
| Where is the nearest hospital? | أين هو المستشفى؟ | ⚠️ drops "nearest" |
| The committee adopted the report without a vote. | اعتمدت اللجنة الفرعية دون تصويت. | ⚠️ drops "the report", adds "sub-" |
| Why do I need to translate this? | لماذا يجب أن أفعل هذا؟ | ⚠️ fluent but changes the meaning |
| Well, back to the fun. | حسنا، عد إلى الخلف | ❌ wrong meaning |
| Put caution in the shotgun seat. | ضع السلاح في السلاح. | ❌ hallucinated (the reference pair in the dataset is also misaligned) |

**Greedy vs beam on the same inputs:**

| English | Greedy | Beam |
|---|---|---|
| Haven't you ever heard that the unexamined life is not worth living? | ألم تسمع ذلك من قبل حياة لا يستحق الحياة التي لا يستحق أن تكون على الإطلاق؟ | ألم تسمع أن الحياة التي لم أسمع بها؟ |
| Why do I need to translate this? | لماذا أحتاج إلى هذا العصا؟ | لماذا يجب أن أفعل هذا؟ |

Greedy decoding repeats "لا يستحق" ("not worth") twice, while beam search with repeat blocking avoids the loop but sometimes shortens or changes the sentence.

---

## Attention analysis

All figures below use one validation sentence and the greedy translation:

> *Haven't you ever heard that the unexamined life is not worth living?*
> → ألم تسمع ذلك من قبل حياة لا يستحق الحياة التي لا يستحق أن تكون على الإطلاق ؟

### Cross-attention: where alignment shows up

![Cross-attention heatmaps](assets/cross_attention.png)

Rows are generated Arabic tokens and columns are English source tokens.

- **Layer 1, heads 0 and 5:** both occurrences of "لا" attend to **"not"**.
- **Layer 2, heads 0 and 4:** both occurrences of "يستحق" attend to **"worth"**.
- Layer 0 cross-attention is diffuse and shows little alignment, so alignment appears in the deeper layers.

This also explains the repetition in the greedy output. Both copies of "لا يستحق" look at the same source words "not worth". A Transformer has no built-in record of which source words were already translated (no coverage mechanism), so it can translate the same words again.

### Decoder self-attention: attention sinks

![Decoder self-attention](assets/decoder_attention.png)

In layer 2, several heads send nearly all of their attention to the first token `[SOS]`: six of the eight heads have an average peak attention of 84–97% per position, and that peak is almost always `[SOS]`. Heads with nothing useful to do seem to park their attention on a fixed position, a known "attention sink" behavior. Layers 0 and 1 are more scattered: their heads spread attention over several earlier tokens or favour a few specific ones.

### How focused are the heads?

![Attention entropy per layer](assets/attention_entropy.png)

Mean attention entropy (lower = more peaked) over the 8 heads of each plotted layer. Cross-attention becomes steadily more focused with depth (1.51 → 1.25 → 0.89), and decoder self-attention in layer 2 is the most peaked (0.42), mostly because of the sink heads. Encoder self-attention stays fairly diffuse and did not show clean, interpretable patterns for this sentence.

> Only layers 0–2 of the 5 layers were plotted, and only for a single sentence, so treat these as illustrations rather than statistics.

---

## Quick start

### 1. Install

```bash
pip install torch tokenizers datasets sacrebleu huggingface_hub tensorboard
```

### 2. Get the code and the model

Clone this repository, then download the weights:

```python
from huggingface_hub import hf_hub_download

weights = hf_hub_download("Sohaila052/English_to_Arabic_translator", "latest.pt")
```

You also need the two tokenizer files (`bpe_tokenizer_en.json`, `bpe_tokenizer_ar.json`, or whatever names your `config.py` uses). They must be the exact tokenizers the model was trained with. In `config.py`, set `DRIVE_DIR` (or `tokenizer_file`) to the folder that contains them.

### 3. Translate

```python
from config import get_config
from decoding import Translator

translator = Translator(get_config(), weights_path=weights)

print(translator.translate("How are you doing today?"))          # beam search (default)
print(translator.translate("I will see you tomorrow.", method="greedy"))
```

Beam-search options:

```python
translator.translate("Where is the nearest hospital?", method="beam",
                     beam_size=4, length_penalty=0.6, no_repeat_ngram=3)
```

### 4. Evaluate BLEU

```python
from train import load_data
from decoding import evaluate_bleu

_, val_dl, _, _, _ = load_data(get_config())
evaluate_bleu(translator, val_dl, n=500, beam_size=4)
```

### 5. Train from scratch (Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/en_ar_transformer
!pip -q install datasets tokenizers tensorboard sacrebleu
!python train.py
```

If the runtime disconnects, run the same cell again: training resumes automatically from `weights_v2/latest.pt`, restarting at the beginning of the unfinished epoch. Make sure the runtime type is **GPU**, otherwise it silently falls back to CPU.

---

## Repository structure

```
.
├── config.py              # hyperparameters and checkpoint paths
├── dataset.py             # BilingualDataset, padding and causal masks
├── model.py               # Transformer implementation
├── train.py               # data loading, tokenizers, training loop, resumable checkpoints
├── decoding.py            # greedy + beam search, Translator class, BLEU evaluation
├── Inference.ipynb        # validation examples and custom sentences
├── Beam_Search.ipynb      # greedy vs beam comparison, BLEU sweeps
├── attention_visual.ipynb # encoder / decoder / cross attention heatmaps
└── assets/                # figures used in this README
```

---

## What went wrong along the way

Building this surfaced several problems. They are kept here because they are the most useful part of the project.

| Problem | Cause | Fix |
|---|---|---|
| Training "worked" but learned nothing useful | Masks passed in the wrong order to `decode()`, and a wrong causal mask that let the decoder see future tokens | Fixed argument order and used `triu(diagonal=1) == 0` |
| First run produced generic phrases and empty outputs | Only 15k pairs, word-level vocabulary, constant LR | Moved to 300k pairs, BPE, warmup and decay |
| Validation crashed or was extremely slow | Greedy decoding ran after every batch, with a validation batch size that violated its own assertion | Validate once per epoch with batch size 1 |
| Run crashed after ~1,100 steps | Pairs were filtered by word count, but the model counts BPE tokens (3 of 261k pairs were too long) | Filter by real token length |
| Resume after a disconnect would have been wrong | Model weights, scheduler and scaler were not all restored; tokenizers and split were not fixed | Atomic checkpoints with full state, seeded split, tokenizers saved on Drive |
| Run silently started on CPU | New Colab runtime had no GPU assigned | Always check the `Using device:` line |
| Greedy outputs looped ("لا ، لا ، لا ...") | No coverage mechanism, greedy decoding | Beam search with repeat blocking |
| Beam search in the reference tutorial dropped finished hypotheses | Finished candidates were skipped when building the next beam | Rewrote it to keep a separate list of finished hypotheses |

---

## Limitations

- **Low quality ceiling:** BLEU ≈ 6 on a noisy single-reference validation set. Meaning is only partly preserved on longer or less common sentences, and the model sometimes drops or invents words.
- **Domain bias:** training data is mostly subtitles and UN text, so formal phrases drift towards UN vocabulary (e.g. "اللجنة الفرعية").
- **Short inputs only:** trained on pairs of at most 40 words and 160 tokens; longer inputs are truncated.
- **Beam search can stop early:** the implementation stops once `beam_size` hypotheses have finished, so some outputs are cut short.
- **Evaluation:** measured on a validation slice of the training sample, not the official OPUS-100 test split.
- **Arabic only in MSA-style written form:** no dialect handling, and diacritics are stripped.

---

## Future work

- Fine-tune a pretrained translation model (e.g. `Helsinki-NLP/opus-mt-en-ar`) and compare it with this from-scratch baseline.
- Train on more data (the full 1M OPUS-100 pairs) and try a larger model and checkpoint averaging.
- Fix the beam-search stopping rule so alive hypotheses can still beat finished ones.
- Evaluate on the official OPUS-100 test split, and add chrF alongside BLEU.
- Add a coverage penalty to reduce repeated or omitted source words.
- Plot attention for layers 3–4 and for many sentences instead of one.

---

## Acknowledgements and references

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017.
- Zhang et al., [*Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation*](https://arxiv.org/abs/2004.11867), 2020 (OPUS-100).
- The code structure and notebooks are inspired by the popular PyTorch Transformer tutorial by Umar Jamil.
- Libraries: PyTorch, Hugging Face `datasets` and `tokenizers`, sacreBLEU, Altair, Matplotlib.

## License

Add your license here (for example MIT).

## Author

[Sohaila052](https://huggingface.co/Sohaila052)
