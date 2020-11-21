
import torch
import torch.optim as optim

import spacy

from CNN import CNN
import dataset as ds 
import params as P

def set_NN(text):

    cnn_model = CNN(len(text.vocab),
                    P.configure()['embedding_dim'],
                    P.configure()['n_filters'],
                    P.configure()['filter_sizes'],
                    P.configure()['output_dim'],
                    P.configure()['dropout'],
                    pad_idx=text.vocab.stoi[text.pad_token])
    print(f'[+] Model Configured...\n \
          {cnn_model}')
    return cnn_model

def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def embed_vectors(text, model):
    
    pretrained = text.vocab.vectors
    
    model.embedding.weight.data.copy_(pretrained)
    print('[+] Pretrained Vectors Set...')
    UNK_IDX = text.vocab.stoi[text.unk_token]
    PAD_IDX = text.vocab.stoi[text.pad_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(P.configure()['embedding_dim'])
    model.embedding.weight.data[PAD_IDX] = torch.zeros(P.configure()['embedding_dim'])
    print('[+] Embedding Weights Set...')

    return model

def fetch_loss_utils(model):

    device = ds.fetch_device()
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    return model, optimizer, criterion

def binary_acc(preds, y):

    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)

    return acc

def train(model, iterator, optimizer, criterion):

    epoch_loss, epoch_acc = 0,0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()

        preds = model(batch.text).squeeze(1)
        print('[+] ----- Foward Pass -> *')
        print(f'[i] Pred Shape: {preds.shape}')
        loss = criterion(preds, batch.label)
        acc = binary_acc(preds, batch.label)

        loss.backward()
        print('[+] * <- Back-Prop -----')
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):

    epoch_loss, epoch_acc = 0,0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            preds = model(batch.text).squeeze(1)
            print('[+] ----- Foward Pass -> *')
            print(f'[i] Pred Shape: {preds.shape}')
            loss = criterion(preds, batch.label)
            acc = binary_acc(preds, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def predict_sentiment(model, text, sentence, min_len = 5):

    nlp = spacy.load('en')

    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]

    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    
    indexed = [text.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(ds.fetch_device())
    tensor = tensor.unsqueeze(0)
    pred = torch.sigmoid(model(tensor))

    if pred < 0.5:

        sentiment = 'Negative'

    else:

        sentiment = 'Positive'

    print(f'Text: {sentence}\n \
            Score: {pred.item()}\n \
            Sentiment: {sentiment}')

    return pred, sentiment