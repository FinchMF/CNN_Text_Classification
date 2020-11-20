

def configure():

    params = {}

    params['sentiments'] = [

        # happy
        'positivity','positivevibes', 'positivethinking', 
        'positivequotes', 'loveyourself', 
        'happy', 'love', 'motivation', 
        # sad
        'depression', 'depressionnap', 'depressionanxiety', 
        'anxiety', 'anxietydepression', 'sad', 'verysad', 
        # afraid
        'afraid', 'fear', 'death', 'scared', 'scarystories', 
        # anger
        'anger', 'rage', 'disgusted', 'mad', 
        'hate', 'shitty', 'angry', 'ridiculous',
        # hopeful
        'hope', 'hopeful', 'wishfulthinking', 
        'wishing', 'joy', 'excited', 'kindness',
        # calm
        'calm', 'reasonable', 'peace', 'peaceful', 'tranquility', 
        'zen', 'meditation', 'blessed'
    ]

    params['sentiment_adjusted'] = [

        'happy',
        'sad',
        'afraid',
        'anger',
        'hopeful',
        'calm'
    ]

    params['SEED'] = 1234
    params['GLOVE_DIR'] = 'glove.twitter.27B.200d'

    params['MAX_SIZE'] = 25_000
    params['BATCH_SIZE'] = 64
    params['EMBEDDING_DIM'] = 200
    params['N_FILTERS'] = 200
    params['FILTER_SIZES'] = [2,3,4]
    params['DROP'] = 0.5

    params['EPOCHS'] = 5
    params['MODEL'] = 'model/CNN_Text-Model.h5'
    params['dataset_dir'] = 'crawler_tweets/'

    return params