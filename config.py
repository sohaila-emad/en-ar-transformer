from pathlib import Path

DRIVE_DIR = "/content/drive/MyDrive/en_ar_transformer"

def get_config():
    return {
        "lang_src": "en",
        "lang_tgt": "ar",
        "seq_len": 160,
        "vocab_size": 16000,
        "max_words": 40,
        "num_samples": 300000,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "warmup_steps": 4000,
        "num_epochs": 12,
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 5,
        "num_decoder_layers": 5,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "experiment_name": f"{DRIVE_DIR}/runs/tmodel",
        "model_folder": f"{DRIVE_DIR}/weights_v2",
        "model_basename": "tmodel_",
        "tokenizer_file": f"{DRIVE_DIR}/bpe_tokenizer_{{0}}.json",
        "keep_last": 1,        # keep only the last N epoch checkpoints (Drive space!)
        "seed": 42,
    }

def get_weights_file_path(config, epoch):
    return str(Path(config["model_folder"]) / f"{config['model_basename']}{epoch}.pt")

def get_latest_weights_path(config):
    return str(Path(config["model_folder"]) / "latest.pt")