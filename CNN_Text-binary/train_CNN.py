
import time

import torch

import dataset as ds 
import params as P
import build

def train_model():

    # prepare data
    text, label, train_data, valid_data, test_data = ds.fetch_data()
    text, label = ds.build_vocabulary(text, label, train_data)
    train_iter, valid_iter, test_iter = ds.fetch_iterators(train_data, valid_data, test_data)   
    # build model and set parameters
    cnn_model = build.set_NN(text)
    print(f'The model has {build.count_parameters(cnn_model):,} trainable parameters')
    cnn_model = build.embed_vectors(text, cnn_model)
    cnn_model, optimizer, criterion = build.fetch_loss_utils(cnn_model)
    # 'save model' conditional
    best_valid_loss = float('inf')
    # training loop
    print('[i] Begin Training...')
    for epoch in range(P.configure()['EPOCHS']):

        start_time = time.time()
        
        train_loss, train_acc = build.train(cnn_model, train_iter, optimizer, criterion)
        valid_loss, valid_acc = build.evaluate(cnn_model, valid_iter, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = build.epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            
            best_valid_loss = valid_loss
            torch.save(cnn_model, P.configure()['model'])

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    
    print('[i] Training Finished...\n')

    return cnn_model, text



if __name__ == '__main__':

    cnn_model, text = train_model()

    print('[i] Evaluate Model\n')
    build.predict_sentiment(cnn_model, text, P.configure()['positive_sentence'])
    build.predict_sentiment(cnn_model, text, P.configure()['negative_sentence'])





        




