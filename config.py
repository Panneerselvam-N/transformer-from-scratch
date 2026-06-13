def get_config():
    config = {
        "batch_size": 16,
        "num_epochs": 100,
        "lr": 1e-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "model_checkpoints",
        "model_basename": "tansformer_translation",
        "preload_model": None,
        "checkpoint_name": "model_best.pth",
        "tokenizer_path": "tokenizer.json",
        "experiment_name": "transformer_translation",
        "dims_ff": 2048,
        "num_heads": 8,
        "num_layers": 8,
        "dropout": 0.1,
        # Set to an integer to limit total dataset size (train+val) for quick debugging on small GPUs.
        # Set to None to use the full dataset.
        "max_dataset_samples": 10000
    }

    return config

def get_weights_file_path(config,epoch):
    return f"{config['model_folder']}/{config['model_basename']}_epoch_{epoch}.pth"