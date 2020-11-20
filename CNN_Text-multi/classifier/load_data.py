
import torch
from torchtext import data, datasets

import params as p


def load_data():

    torch.manual_seed(p.configure()['SEED'])
    torch.backends.cudnn.deterministic = True

    TEXT = data.Field(tokenize='spacy')
    LABEL = data.Field()
    CLASS = data.Field(sequential=False, use_vocab=False)
    
    fields = [(None, None), ('text', TEXT), ('label', LABEL), ('cl', CLASS)]

    train_data, valid_data, test_data = data.TabularDataset.splits(
                                        path = 'data',
                                        train = 'train.csv',
                                        validation = 'valid.csv',
                                        test = 'test.csv',
                                        format = 'csv',
                                        fields = fields,
                                        skip_header = True
                                        )

    return TEXT, LABEL, train_data, valid_data, test_data


def build_vocabulary(TEXT, LABEL, train_data):

    TEXT.build_vocab(train_data,
                    max_size=p.configure()['MAX_SIZE'],
                    vectors=p.configure()['GLOVE_DIR'],
                    unk_init=torch.Tensor.normal_
                    )
    print('[i] Text Vocabulary Built...')

    LABEL.build_vocab(train_data)
    print('[i] Label Vocabulary Built...')

    return TEXT, LABEL


def fetch_device():

    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fetch_iterators(train, valid, test):

    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train, valid, test),
        sort_key= lambda x: x.text,
        batch_size=p.configure()['BATCH_SIZE'],
        device=fetch_device()
    )

    print('[i] Data Loaders Set...')

    return train_iter, valid_iter, test_iter







