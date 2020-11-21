

def configure():

    params = {}
    # parameters
    params['seed'] = 1234
    params['max_vocab_size'] = 25000
    params['batch_size'] = 64
    params['embedding_dim'] = 100
    params['n_filters'] = 100
    params['filter_sizes'] = [3,4,5]
    params['output_dim'] = 1
    params['dropout'] = 0.5
    params['EPOCHS'] = 5
    # model path
    params['model'] = 'model/CNN_Text-Model.h5'
    # evaluation sentence
    params['positive_sentence'] = 'The film was really wonderful'
    params['negative_sentence'] = 'The film was absolutely terrible!'

    return params