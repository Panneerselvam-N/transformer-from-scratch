import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,seq_len ,dropout : float):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # create a dummy positional encoding matrix
        pos_encoding = torch.zeros(seq_len, d_model)
        # create the position indices
        position = torch.arange(0, seq_len).unsqueeze(1)
        # calculate the div_term for sine and cosine functions
        div_term_sin = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        dev_term_cos = torch.exp(torch.arange(1, d_model, 2) * -(math.log(10000.0) / d_model))
        # apply sine and cosine functions
        pos_encoding[:, 0::2] = torch.sin(position * div_term_sin)
        pos_encoding[:, 1::2] = torch.cos(position * dev_term_cos)   

        pos_encoding = pos_encoding.unsqueeze(0)  # add batch dimension

        self.register_buffer('pos_encoding', pos_encoding)
          

    def forward(self, x):
        # add positional encoding to the input tensor
        x =  x + self.pos_encoding[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)
    


class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):

        d_k = query.size(-1)
        # (batch_size, num_heads, seq_len, d_k) x (batch_size, num_heads, d_k, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch_size, num_heads, seq_len, seq_len) x (batch_size, num_heads, seq_len, d_k) -> (batch_size, num_heads, seq_len, d_k)
        return torch.matmul(attention_scores, value), attention_scores

        


    def forward(self, query, key, value, mask=None):    
        query = self.w_q(query) # (batch_size, seq_len, d_model)
        key = self.w_k(key) # (batch_size, seq_len, d_model)
        value = self.w_v(value)  # (batch_size, seq_len, d_model)

        # (bacth_size,seq_len,d_model) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        query = query.view(query.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        # calculate attention scores

        x , attesntion_scores = MultiHeadAttention.attention(query, key, value, mask)

        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model)
        x = self.w_out(x)
        return self.dropout(x), attesntion_scores
    

class ResidualConnection(nn.Module):
        def __init__(self,dropout):
            super().__init__()
            self.norm = LayerNormalization()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))
        

class EncodeBlock(nn.Module):
        def __init__(self, attention : MultiHeadAttention, feed_forward : FeedForward, dropout : float):
            super(EncodeBlock, self).__init__()
            self.attention = attention
            self.feed_forward = feed_forward
            self.residual_coonnction= nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

        def forward(self, x, mask):
            x = self.residual_coonnction[0](x, lambda x: self.attention(x, x, x, mask)[0])
            x = self.residual_coonnction[1](x, self.feed_forward)
            return x
        
class Encoder(nn.Module):
    def __init__(self, layer : nn.ModuleList):
        super(Encoder, self).__init__()
        self.layers = layer
        self.norm = LayerNormalization()    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    


class DecoderBlock(nn.Module):
    def __init__(self, attention : MultiHeadAttention, cross_attention : MultiHeadAttention, feed_forward : FeedForward, dropout : float):
        super(DecoderBlock, self).__init__()
        self.attention = attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.attention(x, x, x, tgt_mask)[0])
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, enc_output, enc_output, src_mask)[0])
        x = self.residual_connection[2](x, self.feed_forward)
        
        return x
    

class Decoder(nn.Module):
    def __init__(self, layer : nn.ModuleList):
        super(Decoder, self).__init__()
        self.layers = layer
        self.norm = LayerNormalization()    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(ProjectionLayer, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim=-1)
    


class Transformer(nn.Module):
    def __init__(self, encoder : Encoder, decoder : Decoder, src_embed : InputEmbedding, tgt_embed : InputEmbedding, pos_encoder : PositionalEncoding, pos_decoder : PositionalEncoding, projection : ProjectionLayer):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.pos_encoder = pos_encoder
        self.pos_decoder = pos_decoder
        self.projection = projection

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.pos_encoder(src)
        return self.encoder(src, src_mask)
    def decode(self, target, enc_output, src_mask, tgt_mask):
        target = self.tgt_embed(target)
        target = self.pos_decoder(target)
        return self.decoder(target, enc_output, src_mask, tgt_mask)
    
    def project_output(self, dec_output):
        return self.projection(dec_output)
    


def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, d_ff=2048, num_heads=8, num_layers=6, dropout=0.1):
    
    # create emmbbing layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # create positional encoding layers
    pos_encoder = PositionalEncoding(d_model, src_seq_len, dropout)
    pos_decoder = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # create encoder layers
    encoder_blocks = []
    for _ in range(num_layers):
        enoder_attention = MultiHeadAttention(d_model, num_heads, dropout)
        encoder_feed_forward = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncodeBlock(enoder_attention, encoder_feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    # create decoder layers
    decoder_blocks = []
    for _ in range(num_layers):
        decoder_attention = MultiHeadAttention(d_model, num_heads, dropout)
        cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        decoder_feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_attention, cross_attention, decoder_feed_forward, dropout)
        decoder_blocks.append(decoder_block)


    # create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create projection layer
    projection = ProjectionLayer(d_model, tgt_vocab_size)
    # create transformer model
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, pos_encoder, pos_decoder, projection)


   # initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer


transfomer = build_transformer(src_vocab_size=10000, tgt_vocab_size=10000, src_seq_len=100, tgt_seq_len=100)