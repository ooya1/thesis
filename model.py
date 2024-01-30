import numpy as np
import pandas as pd
import re
import bz2
from collections import Counter
import nltk
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.utils.data import *

class lstm(nn.Module):
    def __init__(self, vocab_size):
        super(lstm, self).__init__()
        self.n_layers  = 2 
        self.hidden_dim = 256 
        embedding_dim = 400 
        drop_prob=0.3
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            self.hidden_dim,
                            self.n_layers,
                            dropout=drop_prob, 
                            batch_first=True 
                           )

        self.fc = nn.Linear(in_features=self.hidden_dim,
                            out_features=1
                            ) 
        self.sigmoid = nn.Sigmoid() 

        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, x, hidden):

        batch_size = x.size(0) 

        x = x.long() 
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden) 
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        out = out.view(batch_size, -1)
        out = out[:,-1]

        return out, hidden 
    
    def init_hidden(self, batch_size,device):

        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
                 )
        return hidden

    
class roberta(nn.Module):
    def __init__(self, output_dim):
        super(roberta, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(768, output_dim)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x, attention_mask):
        batch_size = x.size(0) 
        context = x 
        mask = attention_mask
        _, pooled = self.bert(context,  attention_mask=mask)
        out = self.fc(pooled)
        out = self.sigmoid(out)
        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out