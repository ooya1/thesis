import numpy as np
import pandas as pd
import re
import bz2
from collections import Counter
import nltk
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.utils.data import *
import random

def pad_input(sentences, seq_len):

    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def predict(sentence,model,device='cuda'):
    sentences = [[word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(sentence)]]
    sentences = pad_input(sentences, 200)

    sentences = torch.Tensor(sentences).long().to(device)

    h = (torch.Tensor(2, 1, 512).zero_().to(device),
         torch.Tensor(2, 1, 512).zero_().to(device))
    h = tuple([each.data for each in h])

    if model(sentences, h)[0] >= 0.5:
        print("positive")
    else:
        print("negative")



def filt(train,test):
    train_sentences = list(train['sentence'].apply(lambda x: x.lower()))
    test_sentences = list(test['sentence'].apply(lambda x: x.lower()))

    for i in range(len(train_sentences)):
        train_sentences[i] = re.sub('\d','0',train_sentences[i])

    for i in range(len(test_sentences)):
        test_sentences[i] = re.sub('\d','0',test_sentences[i])

    for i in range(len(train_sentences)):
        if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:
            train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])

    for i in range(len(test_sentences)):
        if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in test_sentences[i]:
            test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])
    return train_sentences,test_sentences
            
def genDataLoader(sentences,labels,TOKENIZER):

    encoded_input = TOKENIZER(sentences, return_tensors='pt',padding=True,truncation=True,max_length=256)

    data_gen = TensorDataset(encoded_input["input_ids"],
                             encoded_input["attention_mask"],
                             torch.from_numpy(labels))

    sampler = RandomSampler(data_gen)
    loader = DataLoader(data_gen, sampler=sampler, batch_size=16)
    return loader
        
        
def train_roberta(model, device, train_loader, test_loader, optimizer,savepath,epoch):
    model.train()
    criterion = nn.BCELoss().to(device)
    for epoch in range(1, epoch + 1):
        batch_idx = 0
        for (x1, x2,y) in tqdm(train_loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device).reshape(-1,1)
            y_pred = model(x1,attention_mask = x2,)
            optimizer.zero_grad()
            loss = criterion(y_pred, y.squeeze().float()) 
            loss.backward()
            optimizer.step()
            batch_idx += 1
            if(batch_idx + 1) % 100 == 0: 
                print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx+1) * len(x1),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               loss.item()))
    torch.save(model.state_dict(), savepath)

def test_roberta(model, device, test_loader):

    model.eval()
    test_loss = 0.0
    acc = 0
    criterion = nn.BCELoss().to(device)
    for (x1, x2,y) in tqdm(test_loader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device).reshape(-1,1)
        with torch.no_grad():
            y_ = model(x1,attention_mask =x2,)
        test_loss += criterion(y_, y.squeeze().float())
        y_[y_>=0.5]=1
        y_[y_<=0.5]=0
        acc += torch.sum(y_==y.reshape(-1))
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
          test_loss, acc, len(test_loader.dataset),
          100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True