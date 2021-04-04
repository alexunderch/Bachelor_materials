import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import numpy as np
import time

def scaled_dot_product(q, k, v, mask = None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -1e16)
    attention = F.softmax(attn_logits, dim = -1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        """Just almost a copy of torch.nn.MultiheadAttention"""
        super(type(self), self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 2 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()


    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)


    def forward(self, q, c, mask = None, return_attention_weights = False):
        batch_size, seq_length, embed_dim = q.size()
        kv = self.qkv_proj(c)

        # Separate Q, K, V from linear output
        kv = kv.reshape(c.size(0), c.size(1), self.num_heads, 2 * self.head_dim)
        kv = kv.permute(0, 2, 1, 3) # [batch_size, seq_len, n_heads, dims]
        k, v = kv.chunk(2, dim = -1)
        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask = mask)
        values = values.permute(0, 2, 1, 3) # [batch_size, seq_len, n_heads, dims]
        values = values.reshape(c.size(0), seq_length, embed_dim)
        out = self.o_proj(values)

        if return_attention_weights: return out, attention
        return out

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(
            num_embeddings = input_dim,
            embedding_dim = emb_dim)

        self.self_attn = MultiheadAttention(emb_dim, emb_dim, 4)

        self.relu = nn.ReLU()
        self.rnn = nn.LSTM(
            input_size = emb_dim,
            hidden_size = hid_dim,
            num_layers = n_layers,
            dropout = dropout)
        
        self.norm1 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, src, mask = None):
        
        #src = [src sent len, batch size]
        
        # Compute an embedding from the src data and apply dropout to it
        embedded = self.embedding(src)# <YOUR CODE HERE>
        attn_out = self.self_attn(embedded.transpose(0, 1), embedded.transpose(0, 1))
        embedded = embedded + self.dropout(attn_out.transpose(0, 1))
        embedded = self.norm1(embedded)
        
        output, (hidden, cell) = self.rnn(embedded) 

        #embedded = [src sent len, batch size, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        # print(output.shape, hidden.shape, cell.shape)
        return output, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(
            num_embeddings = output_dim,
            embedding_dim = emb_dim)
        
        self.self_attention = MultiheadAttention(emb_dim, emb_dim, 4)

        self.encoder_attention = MultiheadAttention(hid_dim,  hid_dim, 4)

        self.relu1 = nn.ReLU()
        self.rnn = nn.LSTM(
            input_size = emb_dim,
            hidden_size = hid_dim,
            num_layers = n_layers,
            dropout = dropout)

        self.relu2 = nn.ReLU()
        self.out = nn.Linear(
            in_features = hid_dim,
            out_features = output_dim)
        self.dropout = nn.Dropout(p = dropout)
        self.dropout1 = nn.Dropout(p = dropout)
        self.dropout2 = nn.Dropout(p = dropout)
        self.norm1 = nn.LayerNorm(hid_dim)
    def forward(self, _input, hidden, cell, encoder_outputs):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        encoded_input = _input.unsqueeze(0)
        
        #input = [1, batch size]
        
        # Compute an embedding from the input data and apply dropout to it
        embedded = self.dropout(self.embedding(encoded_input))
        self_attn_out = self.self_attention(embedded.transpose(0, 1), 
                                            embedded.transpose(0, 1))
        embedded = embedded + self.dropout1(self_attn_out.transpose(0, 1)) 


        #embedded = [1, batch size, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #sent len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell)) 
        output =  output + self.dropout2(self.encoder_attention(output.transpose(0, 1), 
                                                                encoder_outputs.transpose(0, 1)).transpose(0, 1))
#         output = self.norm1(output)
        prediction = self.out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg   sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_outputs, hidden, cell = self.encoder(src)
        #first input to the decoder is the <sos> tokens
        _input = trg[0, :]
        
        for t in range(1, max_len):
            
            output, hidden, cell = self.decoder(_input, hidden, cell, enc_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            _input = (trg[t] if teacher_force else top1)
        
        return outputs

def positional_encoding(position, d_model, device):
    angles = np.arange(position)[:, None] *  1. / np.power(10000, (2 * (np.arange(d_model)[None, :] // 2)) / np.float32(d_model)) 
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return torch.FloatTensor(angles).unsqueeze(0).to(device)

class Encoder_CNN(nn.Module):
    def __init__(self, device, input_dim, emb_dim, hid_dim, output_dim, use_pos_encoding: bool, 
                 kernel_size: int, padding: int, n_layers: int, 
                 dropout: float):
        super(type(self), self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.device = device
        self.use_pos_encoding = use_pos_encoding
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.s_conv_block = [nn.Conv1d(emb_dim, hid_dim, kernel_size, padding = kernel_size // 2),
                                          nn.BatchNorm1d(hid_dim), 
                                          nn.Dropout(dropout),
                                          # nn.AvgPool1d(kernel_size)
                                          ]
        self.c_conv_block = [nn.Conv1d(hid_dim, hid_dim, kernel_size, padding = kernel_size // 2),
                                          nn.BatchNorm1d(hid_dim), 
                                          nn.Dropout(dropout)]
        modules = self.s_conv_block
        for _ in range(n_layers - 1): modules.extend(self.c_conv_block)
        self.conv_net = nn.ModuleList(modules)
        self.relu = nn.ReLU()

        if output_dim is not None: self.lin_out = nn.Linear(hid_dim, output_dim)
        else:
            self.lin_out = None
            self.output_dim = hid_dim
        
        self.dropout = nn.Dropout(dropout)
        self.ot_pooling = nn.AdaptiveMaxPool1d(1)
        self.flatten = torch.nn.Flatten()

    def forward(self, src):
        embedded = self.embedding(src)         
        if self.use_pos_encoding:
            pos_encoding_emb = torch.zeros([src.shape[1], src.shape[0], self.emb_dim], dtype = torch.float).to(self.device)
            pos_encoding_emb = positional_encoding(src.shape[0], self.emb_dim, self.device)
            embedded = embedded + pos_encoding_emb.permute(1, 0, 2)

        embedded = embedded.permute((1, 2, 0))
        for  layer in self.conv_net: 
            embedded = self.dropout(self.relu(layer(embedded)))

        output = self.flatten(self.ot_pooling(embedded))
        if self.lin_out: output = self.lin_out(output)
        return output  

class CNN2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        # assert encoder.n_layers == decoder.n_layers, \
        #     "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        dec_hid_size = self.decoder.hid_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_output = self.encoder(src).contiguous().repeat(self.decoder.n_layers, 1, 1)
        encoder_outputs = self.encoder(src).contiguous().repeat(self.decoder.n_layers, 1, 1)

        
        #first input to the decoder is the <sos> tokens
        _input = trg[0, :]
              
        for t in range(1, max_len):
            if t == 1:
                # output, hidden = self.decoder(_input, enc_output.contiguous().view(self.decoder.n_layers, batch_size, 2 * dec_hid_size))
                output, hidden = self.decoder(_input, enc_output, encoder_outputs)                      
            else: output, hidden = self.decoder(_input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs

class Decoder_CNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(
            num_embeddings = output_dim,
            embedding_dim = emb_dim)
        self.self_attention = MultiheadAttention(emb_dim, emb_dim, 4)
        self.encoder_attention = MultiheadAttention(hid_dim,  hid_dim, 4)
        self.rnn = nn.GRU(
            input_size = emb_dim,
            hidden_size = hid_dim,
            num_layers = n_layers,
            dropout = dropout)

        
        self.out = nn.Linear(
            in_features =  hid_dim,
            out_features = output_dim)
        
        self.dropout = nn.Dropout(p = dropout)
        self.dropout1 = nn.Dropout(p = dropout)
        
    def forward(self, _input, hidden, encoder_outputs):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        _input = _input.unsqueeze(0)
        
        #input = [1, batch size]
        
        # Compute an embedding from the input data and apply dropout to it
        embedded = self.embedding(_input)
        self_attn_out = self.self_attention(embedded.transpose(0, 1), 
                                            embedded.transpose(0, 1))
        embedded = embedded + self.dropout1(self_attn_out.transpose(0, 1)) 
        #embedded = [1, batch size, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #sent len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        
        output, hidden = self.rnn(embedded, hidden)
        output =  output + self.dropout(self.encoder_attention(output.transpose(0, 1), 
                                                                encoder_outputs.transpose(0, 1)).transpose(0, 1))
        prediction = self.out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden
