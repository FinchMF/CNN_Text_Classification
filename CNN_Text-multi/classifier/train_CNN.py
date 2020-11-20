
import time
import dill
import torch

import load_data as ld 
import params as p 
import build


def train_CNN():
    print('[+] Load Data')
    text, label, train_data, valid_data, test_data = ld.load_data()
    print('[+] Build Vocabulary')
    text, label = ld.build_vocabulary(text, label, train_data)
    print('[+] Set Iterators')
    train_iter, valid_iter, test_iter = ld.fetch_iterators(train_data, valid_data, test_data)
    print('[i] Train Iterator Info: \n')
    print(f'[i] Length of Train Iter: {len(train_iter)}')
   
    cnn_model = build.set_NN(text, label)

    print(f'[i] The model has {build.count_parameters(cnn_model):,} trainable parameters')

    cnn_model = build.embed_vectors(text, cnn_model)

    print('[+] Save Text Data')
    with open('model/TEXT.Field', 'wb') as f:
        dill.dump(text, f)

    cnn_model, optimizer, criterion = build.fetch_loss_utils(cnn_model)

    best_valid_loss = float('inf')

    for epoch in range(p.configure()['EPOCHS']):

        start = time.time()

        train_loss, train_acc = build.train(cnn_model, train_iter, optimizer, criterion)
        valid_loss, valid_acc = build.evaluate(cnn_model, valid_iter, criterion)

        end = time.time()

        epoch_mins, epoch_secs = build.epoch_times(start, end)

        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss

            torch.save(cnn_model, p.configure()['MODEL'])

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


    return cnn_model



if __name__ == '__main__':

    cnn_model = train_CNN()

    print('[+] Reload Text Data')
    with open('model/TEXT.Field', 'rb') as f:
        text = dill.load(f)

    print('[+] Generate Test...')
    test_data = build.gen_test()
    
    print(f'[i] Tweet: {test_data.Tweet}')
    print(f'[i] Actual Class: {test_data.Class}')
    print(f'[i] Predicted Class: {build.predict_sentiment(cnn_model, text, str(test_data.Tweet))}')





