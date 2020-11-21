
import random
import numpy as np

import torch
from torchtext import data
from torchtext import datasets

import params as P

def fetch_data():

    random.seed(P.configure()['seed'])
    np.random.seed(P.configure()['seed'])
    torch.manual_seed(P.configure()['seed'])
    torch.backends.cudnn.deterministic = True
    print('[+] Seeds Set...')

    text = data.Field(tokenize='spacy', batch_first = True)
    print('[+] Text Recieved...')
    label = data.LabelField(dtype=torch.float)
    print('[+] Label Recieved...')
    print('[+] Transforming...')
    train_data, test_data = datasets.IMDB.splits(text, label)
    print('[+] Train | Test Split Set...')
    train_data, valid_data = train_data.split(random_state=random.seed(P.configure()['seed']))
    print('[+] Train | Validation Split Set...')

    return text, label, train_data, valid_data, test_data

def build_vocabulary(text, label, train_data):

    text.build_vocab(train_data,
                    max_size=P.configure()['max_vocab_size'],
                    vectors='glove.6B.100d',
                    unk_init=torch.Tensor.normal_)
    print('[+] Text Vocabulary Built...')

    label.build_vocab(train_data)
    print('[+] Label Vocabulary Built...')
    
    return text, label

def fetch_device():

    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fetch_iterators(train, valid, test):

    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
                                        (train, valid, test),
                                        batch_size=P.configure()['batch_size'],
                                        device=fetch_device()
                                        )
    print('[+] Dataloaders Set...')
    
    return train_iter, valid_iter, test_iter




        
