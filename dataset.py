import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Any

class BilingualDataset(Dataset):
    def __init__(
        self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # Pass plain strings directly into token_to_id
        # Use torch.tensor (lowercase t) with dtype=torch.int64
        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)



    def __getitem__(self, idx: Any) ->Any:
        # Retrieve the raw item dictionary from random_split or HF dataset
        raw_item = self.ds[idx]
    
        # Check if 'translation' exists (standard Hugging Face MT structure)
        if "translation" in raw_item:
            src_tgt_pair = raw_item["translation"]
        else:
            src_tgt_pair = raw_item

        src_text = src_tgt_pair[self.src_lang]
        tgt_text = src_tgt_pair[self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) -2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) -1

        if dec_num_padding_tokens < 0 or enc_num_padding_tokens < 0:
            raise ValueError(f"Source or target text is too long for the specified sequence length of {self.seq_len}.")
        #add EOS and SOS to the encoder input
        encoder_input = torch.cat([self.sos_token, torch.tensor(enc_input_tokens, dtype=torch.int64), self.eos_token, torch.tensor([self.pad_token.item()] * enc_num_padding_tokens, dtype=torch.int64)])
        #add SOS to the decoder input 
        decoder_input = torch.cat([self.sos_token, torch.tensor(dec_input_tokens, dtype=torch.int64), torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64)])
        #add EOS to the labels(what we expect the decoder to output)
        labels = torch.cat([torch.tensor(dec_input_tokens, dtype=torch.int64), self.eos_token, torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64)])

        assert encoder_input.shape[0] == self.seq_len, f"Encoder input length {encoder_input.shape[0]} does not match the specified sequence length of {self.seq_len}."
        assert decoder_input.shape[0] == self.seq_len, f"Decoder input length {decoder_input.shape[0]} does not match the specified sequence length of {self.seq_len}."
        assert labels.shape[0] == self.seq_len, f"Labels length {labels.shape[0]} does not match the specified sequence length of {self.seq_len}."

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_attention_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1,seq_len)
            "labels": labels,
            "decoder_attention_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()& causal_mask(decoder_input.size(0)) , #(1,seq_len)&(seq_len,seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int32)
    return mask == 0