# Transformers (From Scratch)

This repository contains a minimal implementation of a Transformer-based sequence-to-sequence model built from scratch using PyTorch.

## What it includes

- Custom Transformer implementation (`model.py`) with encoder/decoder blocks and attention.
- Tokenizer training and usage via the `tokenizers` library.
- Dataset handling via `datasets` and a custom `BilingualDataset` wrapper.
- Training loop, validation decoding, and checkpoint saving in `train.py`.

## Quickstart

1. Install dependencies (see `requirements.txt`).
2. Configure training in `config.py`.
3. Run training:

```bash
python3 train.py
```

## Notes

- The script can limit dataset size for small GPUs using `config["max_dataset_samples"]`.
- Validation uses greedy decoding and prints sample outputs.

