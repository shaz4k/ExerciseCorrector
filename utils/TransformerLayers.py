import math
import torch
import torch.nn as nn


class TransformerV1(nn.Module):
    # Encoder Only
    def __init__(self, feature_size, d_model, nhead, num_layers, d_ff=2048, dropout=0.1, max_len=57):
        super(TransformerV1, self).__init__()
        self.linear = nn.Linear(feature_size, d_model)
        self.bn1 = nn.BatchNorm1d(max_len * d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.encoder = TransformerEncoder(d_model, nhead, num_layers, d_ff, dropout)
        self.linear_out = nn.Linear(d_model, feature_size)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        y = self.linear(src)
        y = self.positional_encoding(y)
        output = self.encoder(y)
        output = self.linear_out(output)
        return output + src


class TransformerV2(nn.Module):
    # Encoder and Decoder
    def __init__(self, feature_size, d_model, nhead, num_layers, d_ff=2048, dropout=0.1, max_len=57):
        super(TransformerV2, self).__init__()
        # num_enc_layers # num_dec_layers
        self.linear = nn.Linear(feature_size, d_model)
        self.bn1 = nn.BatchNorm1d(feature_size * d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.encoder = TransformerEncoder(d_model, nhead, num_layers, d_ff, dropout)
        self.decoder = TransformerDecoder(d_model, nhead, num_layers, d_ff, dropout)
        self.linear_out = nn.Linear(d_model, feature_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # embedding layers
        src = self.linear(src)
        tgt = self.linear(tgt)

        # positional encoding
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # encoder
        memory = self.encoder(src, src_mask=src_mask)

        # decoder
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        # linear out
        out = self.linear_out(output)


        return out


class TransformerV3(nn.Module):
    # Encoder and Decoder
    def __init__(self, feature_size, d_model, nhead, num_layers, d_ff=2048, dropout=0.1, max_len=57):
        super(TransformerV3, self).__init__()
        # num_enc_layers # num_dec_layers
        self.linear = nn.Linear(feature_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.encoder = TransformerEncoder(d_model, nhead, num_layers, d_ff, dropout)
        self.decoder = TransformerDecoder(d_model, nhead, num_layers, d_ff, dropout)
        self.linear_out = nn.Linear(d_model, feature_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # embedding layers
        src = self.linear(src)
        tgt = self.linear(tgt)

        # positional encoding
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # encoder
        memory = self.encoder(src, src_mask=src_mask)

        # decoder
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        return output

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)

        attn_output, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(attn_output)
        tgt = self.norm2(tgt)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(ff_output)
        tgt = self.norm3(tgt)

        return tgt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=25):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
