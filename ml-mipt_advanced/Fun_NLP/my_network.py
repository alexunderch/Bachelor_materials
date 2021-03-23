import torch
import torch.nn as nn
import torch.optim as optim

import random
import math
import time

# from transformers import BertTokenizer, BertModel   

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        ####CNN ENCODER SHOULD BE HERE####
        self.embedding = nn.Embedding(
            num_embeddings = input_dim,
            embedding_dim = emb_dim)

        self.conv1 = nn.Conv1d(in_channels = emb_dim, out_channels = 2 * emb_dim, kernel_size = 3, padding = 1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(3 * hid_size)
        self.amp = nn.AdaptiveMaxPool1d(4)      

        self.self_attention = nn.MultiheadAttention(emb_dim, 2)

        self.rnn = nn.LSTM(
            input_size = emb_dim,
            hidden_size = hid_dim,
            num_layers = n_layers,
            dropout = dropout,
            bidirectional = bidirectional)
        
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        
        # Compute an embedding from the src data and apply dropout to it
        embedded = self.embedding(src)# <YOUR CODE HERE>
        embedded = self.conv1(embedded)
        embedded = self.relu1(embedded)
        embedded = self.dropout1(embedded)

        print(embedded.shape)

        output, (hidden, cell) = self.rnn(embedded)
        #embedded = [src sent len, batch size, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return output, hidden, cell

# class Encoder(nn.Module):
#     def __init__(self):
#         """
#         """
#         super(type(self), self).__init__()
        
#         self.model = BertModel.from_pretrained('bert-base-uncased')
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     def forward(self, text: list, max_length = 512, truncate = True, use_padding = True):
#         """
#         """
#         with torch.no_grad():
#             encoded_input = self.tokenizer(text, return_tensors = 'pt', max_length = max_length,
#                                       truncation = truncate, padding = 'max_length')
#         input_ids = encoded_input['input_ids']
#         attention_mask = encoded_input['attention_mask']

#         embeddings = self.model(input_ids, attention_mask = attention_mask, output_hidden_states = True)
#         return embeddings.hidden_states # 13 (batch_size, sequence_length, hidden_size)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(
            num_embeddings = output_dim,
            embedding_dim = emb_dim)
        
        self.encoder_attention = nn.MultiheadAttention(emb_dim, 4)

        self.rnn = nn.LSTM(
            input_size = emb_dim,
            hidden_size = hid_dim,
            num_layers = n_layers,
            dropout = dropout
            bidirectional = bidirectional)
        
        self.out = nn.Linear(
            in_features = hid_dim,
            out_features = output_dim)
        
        self.dropout = nn.Dropout(p = dropout)

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
        embedded = self.dropout(self.embedding(encoded_input))# <YOUR CODE HERE>
        attn_scores, _  = self.encoder_attention(embeded, 
                                        encoder_outputs, 
                                        encoder_outputs)
        #embedded = [1, batch size, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        # <YOUR CODE HERE>
        
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #sent len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        
        output, (hidden, cell) = self.rnn(attn_scores, (hidden, cell))
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
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0, :]
        
        for t in range(1, max_len):
            
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs
