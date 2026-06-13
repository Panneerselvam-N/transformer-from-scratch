import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from datasets import load_dataset
from dataset import BilingualDataset
from model import build_transformer
from config import get_config, get_weights_file_path


def causal_mask(size):
    # Return an integer mask (0/1) with 1s in the lower triangle (causal positions)
    mask = torch.triu(torch.ones(size, size), diagonal=1).to(torch.int)
    return mask==0


def greedy_decode(model,source,source_mask,tokenizer_src,tokenizer_tgt,max_len,device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty((1,1)).fill_(sos_idx).type_as(source).to(device)  # Start with SOS token


    while True:

        if decoder_input.size(1) == max_len:
            break
            
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)  # [1, 1, tgt_seq_len, tgt_seq_len]
        out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)  # [1, tgt_seq_len, d_model]

        probs = model.project_output(out[:,-1])
        _, next_word = torch.max(probs, dim=1)

        decoder_input = torch.cat([decoder_input, torch.empty((1,1)).fill_(next_word.item()).type_as(source).to(device)], dim=1)

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model,validation_ds,tokenizer_src,tokenizer_tgt,max_len,device,print_msg,num_of_sample):
    model.eval()
    count = 0
    with torch.no_grad():
        for batch in validation_ds:
            encoder_input = batch["encoder_input"].to(device)  # [batch_size, seq_len]
            encode_mask = batch["encoder_mask"].to(device)  # [batch_size, 1, 1, seq_len]

            assert encoder_input.shape[0] == 1, "Batch size must be 1 for validation"
            # call greedy decode function to get the predicted sequence
            model_out = greedy_decode(model, encoder_input, encode_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_txt = batch["src_text"][0]
            target_txt = batch["tgt_text"][0]
            predicted_txt = tokenizer_tgt.decode(model_out.cpu().numpy())

            print_msg("-" * 50 )
            print_msg(f"Source: {source_txt}")
            print_msg(f"Target: {target_txt}")
            print_msg(f"Predicted: {predicted_txt}")

            if count == num_of_sample:
                break
            count += 1

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = config["tokenizer_path"].format(lang)

    if not os.path.exists(tokenizer_path):
        print(f"Building tokenizer for {lang}...")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(tokenizer_path)
    else:
        print(f"Loading tokenizer for {lang} from {tokenizer_path}...")
        tokenizer = Tokenizer.from_file(tokenizer_path)

    return tokenizer


def get_dataset(config):

    data_raw = load_dataset(
        "Helsinki-NLP/opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train"
    )

    # Optionally limit dataset size for small-GPU experiments
    if config.get("max_dataset_samples") is not None:
        max_samples = int(config["max_dataset_samples"])
        if max_samples <= 0:
            raise ValueError("max_dataset_samples must be a positive integer or None")
        print(f"Limiting dataset to {max_samples} samples (train+val) to reduce memory usage")
        data_raw = data_raw.select(range(max_samples))

    tokenizer_src = get_or_build_tokenizer(config, data_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, data_raw, config["lang_tgt"])
    train_size = int(len(data_raw) * 0.9)
    val_size = len(data_raw) - train_size
    data_train, data_val = torch.utils.data.random_split(
        data_raw, [train_size, val_size]
    )

    train_dataset = BilingualDataset(
        data_train,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_dataset = BilingualDataset(
        data_val,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    max_src_seq_len = 0
    max_tgt_seq_len = 0
    for item in data_raw:
        src_token_id = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_token_id = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_src_seq_len = max(max_src_seq_len, len(src_token_id))
        max_tgt_seq_len = max(max_tgt_seq_len, len(tgt_token_id))
    print(f"Max source sequence length: {max_src_seq_len}")
    print(f"Max target sequence length: {max_tgt_seq_len}")

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,num_workers=2,pin_memory=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,num_workers=2,pin_memory=True
    )

    return train_data_loader, val_data_loader, tokenizer_src, tokenizer_tgt


def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=config["seq_len"],
        tgt_seq_len=config["seq_len"],
        d_model=config["d_model"],
        d_ff=config["dims_ff"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    return model



def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if config["preload_model"] == "latest":
        best_path = os.path.join(config['model_folder'], config["checkpoint_name"])
        if os.path.exists(best_path):
            config["preload_model"] = "best"
        else:
            config["preload_model"] = None

    if os.path.exists(config["model_folder"]) == False:
        os.mkdir(config["model_folder"])

    train_data_loader, val_data_loader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],eps=1e-9)
    initial_epoch = 0
    global_step = 0
    best_loss = float('inf')
    if config["preload_model"] is not None:
        pt_path = os.path.join(config["model_folder"], config["checkpoint_name"])
        checkpoint = torch.load(pt_path, map_location=device)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint.get('loss', float('inf'))
            initial_epoch = checkpoint.get('epoch', 0)
        else:
            model.load_state_dict(checkpoint)
        print(f"Resumed from epoch {initial_epoch} with best loss {best_loss}")

    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"),label_smoothing=0.1).to(device)

   

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_data_loader:
            model.train()
            encoder_input = batch["encoder_input"].to(device) # [batch_size, seq_len]
            decoder_input = batch["decoder_input"].to(device) # [batch_size, seq_len]
            encoder_mask = batch["encoder_mask"].to(device) # [batch_size, 1, 1, seq_len]
            decoder_mask = batch["decoder_mask"].to(device) # [batch_size, 1, seq_len, seq_len]
            labels = batch["labels"].to(device) # [batch_size, seq_len]

            optimizer.zero_grad()  # Clear previous gradients
            
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            projected_output = model.project_output(decoder_output) # [batch_size, seq_len, tgt_vocab_size]

            loss = loss_fn(projected_output.view(-1,tokenizer_tgt.get_vocab_size()), labels.view(-1))
            total_loss += loss.item()
            batch_count += 1

            loss.backward()  # Compute gradients
            optimizer.step()  # Update parameters


            global_step += 1

        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{config['num_epochs']} - Avg Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }
            model_save_path = os.path.join(config["model_folder"], config["checkpoint_name"])
            torch.save(checkpoint, model_save_path)
            print(f"Best model checkpoint saved to {model_save_path} with loss {best_loss:.4f}")
        if epoch%5 == 0:
          run_validation(model, val_data_loader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, print, num_of_sample=1)

    print(f"Training completed!")


if __name__ == "__main__":
    config = get_config()
    train_model(config)