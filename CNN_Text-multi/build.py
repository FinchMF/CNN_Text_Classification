
import torch
import torch.optim as optim

import spacy
nlp = spacy.load('en')
import pandas as pd 

from CNN import CNN
import load_data as ld
import params as p 

from transform_data import preprocess_tweet

def set_NN(text, label):

    cnn_model = CNN(
                    len(text.vocab),
                    p.configure()['EMBEDDING_DIM'],
                    p.configure()['N_FILTERS'],
                    p.configure()['FILTER_SIZES'],
                    len(label.vocab),
                    p.configure()['DROP'],
                    text.vocab.stoi[text.pad_token]
              )
    print('[i] Model Configured...')
    return cnn_model

def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def embed_vectors(text, model):

    pretrained = text.vocab.vectors

    model.embeds.weight.data.copy_(pretrained)
    print('[i] GloVe Vectors Set...')
    UNK_IDX = text.vocab.stoi[text.unk_token]
    PAD_IDX = text.vocab.stoi[text.pad_token]

    model.embeds.weight.data[UNK_IDX] = torch.zeros(p.configure()['EMBEDDING_DIM'])
    model.embeds.weight.data[PAD_IDX] = torch.zeros(p.configure()['EMBEDDING_DIM'])
    print('[i] Embedding Dimensions Set...')

    return model

def fetch_loss_utils(model):

    device = ld.fetch_device()
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    return model, optimizer, criterion

def categorical_acc(preds, y):

    max_preds = preds.argmax(dim=1, keepdim=True)
    correct = max_preds.squeeze(1).eq(y)

    return correct.sum() / torch.FloatTensor([y.shape[0]])


def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()
        print('---- FORWARD ----> *')
        preds = model(batch.text)
        print(' - Error Assessed - ')

        loss = criterion(preds, batch.cl)

        acc = categorical_acc(preds, batch.cl)

        loss.backward()
        print('* <---- BACK PROP ----')
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    for batch in iterator:
        print('---- FORWARD ----> *')
        preds = model(batch.text)
        print(' - Error Assessed -')

        loss = criterion(preds, batch.cl)

        acc = categorical_acc(preds, batch.cl)

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_times(start, end):

    elapsed_time = end - start
    elapsed_min = int(elapsed_time / 60)
    elasped_secs = int(elapsed_time - (elapsed_min - 60))

    return elapsed_min, elasped_secs


def gen_test():

    data = pd.read_csv('data/test.csv')
    test = data.sample()
    return test


def predict_sentiment(model, text, tweet, min_len=4):
    
    model.eval()
    tweet = preprocess_tweet(tweet)
    tokenized = [tok.text for tok in nlp.tokenizer(tweet)]

    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    
    indexed = [text.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(ld.fetch_device())
    tensor = tensor.unsqueeze(1)
    preds = model(tensor)
    max_preds = preds.argmax(dim=1)

    return max_preds.item()





