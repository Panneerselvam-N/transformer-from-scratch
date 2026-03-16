import torch.nn as nn
from torch.utils.data import Dataset
import torch

class BilingualDataset(Dataset):
    def __init__(self,dataset,tokenizer_src,tokenizer_tgt,lang_src,lang_tgt,seq_len):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len

        self.pad_token = tokenizer_tgt.token_to_id("[PAD]")
        self.eos_token = tokenizer_tgt.token_to_id("[EOS]")
        self.sos_token = tokenizer_tgt.token_to_id("[SOS]")

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        src_target_pair = self.dataset[idx]['translation']
        src_text = src_target_pair[self.lang_src]
        tgt_text = src_target_pair[self.lang_tgt]

        src_encoding = self.tokenizer_src.encode(src_text).ids
        tgt_encoding = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_paddings = self.seq_len - len(src_encoding) - 2
        dec_num_paddings = self.seq_len - len(tgt_encoding) - 1

        if enc_num_paddings < 0 or dec_num_paddings < 0:
            raise ValueError(f"Sequence length {self.seq_len} is too short for the given sentence pair.")
        
        encoder_input = torch.cat([torch.Tensor([self.sos_token]), 
                                   torch.Tensor(src_encoding), 
                                   torch.Tensor([self.eos_token]), 
                                   torch.full((int(enc_num_paddings),), self.pad_token)])
        
        decoder_input = torch.cat([torch.Tensor([self.sos_token]), 
                                   torch.Tensor(tgt_encoding), 
                                   torch.full((int(dec_num_paddings),), self.pad_token)])
        
        labels = torch.cat([torch.Tensor(tgt_encoding), 
                            torch.Tensor([self.eos_token]), 
                            torch.full((int(dec_num_paddings),), self.pad_token)])
        return {
            "encoder_input": encoder_input.long(),
            "decoder_input": decoder_input.long(),
            "encoder_mask": (encoder_input != self.pad_token).long().unsqueeze(0).unsqueeze(0),
            "decoder_mask": (decoder_input != self.pad_token).long().unsqueeze(0) & causal_mask(self.seq_len).unsqueeze(0),
            "labels": labels.long(),
            "src_text": src_text,
            "tgt_text": tgt_text
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((size, size)), diagonal=1).type(torch.int)
    return (mask == 0)