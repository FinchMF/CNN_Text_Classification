
from collections import OrderedDict

import re
import glob
import string

import numpy as np 
import pandas as pd 

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

import spacy
import en_core_web_sm
nlp_pos_model = en_core_web_sm.load()




def fetch_sentiment_datasets(dataset_dir):
    print('[+] Retrieving List of Emotional Datasets...')
    return glob.glob(f'{dataset_dir}/*.csv')


def combine_datasets(dataset_dir, classes):

    sentiment_datasets = fetch_sentiment_datasets(dataset_dir)
    datasets = []
    count = 0
    print('[+] Combining Collected Datasets...')
    for f in sentiment_datasets:
        ds = pd.read_csv(f, lineterminator='\n')
        ds = ds.drop('Unnamed: 0', axis=1)
        ds['Label'] = classes[count]
        ds.rename(columns={'0': 'Tweet'}, inplace=True)
        datasets.append(ds)
        count += 1

    complete_dataset = pd.concat(datasets, ignore_index=True)
    print('[+] Datasets Combined')
    return complete_dataset


def validate_tweet(text):   

    tweet_pos = []
    tweet_val = []
    tweet = nlp_pos_model(text)
    
    for token in tweet:

        pos = {

            'Text': token.text,
            'Lemma': token.lemma_,
            'POS': token.pos_,
            'TAG': token.tag_,
            'DEP': token.dep_
        }

        tweet_pos.append(pos)
        tweet_val.append(pos['POS'])
    print(tweet_val)

    noun = 'NOUN' in tweet_val
    verb = 'VERB' in tweet_val

    if noun and verb:
        print('[i] VALID tweet')
        print(text)
        print(tweet_val)
        return text

    else:
        print('[i] NOT VALID tweet')
        print(text)
        print(tweet_val)
        return 'Not a Valid Tweet'


def fetch_valid_tweets(dataset):

    x = 1
    for idx, row in dataset.iterrows():
        print('[+] validating...')
        row['Tweet'] = validate_tweet(row['Tweet'])
        print(f'[i] Progress: {(x / len(dataset)) * 100}%')
        x += 1
        
    valid_dataset = dataset[dataset.Tweet != 'Not a Valid Tweet']
    print('[i] Dataset Validated')

    return valid_dataset

def preprocess_tweet(text):

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    clean_tw = re.sub('(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', text)

    clean_tw = [char for char in clean_tw if char not in string.punctuation]
    clean_tw = ''.join(clean_tw)
    clean_tw = clean_tw.lower()

    clean_tw = re.sub('(#[A-Za-z0-9_]+)', '\1', clean_tw)

    clean_tw = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', clean_tw)
    clean_tw = re.sub('http\S+', '', clean_tw)

    clean_tw = word_tokenize(clean_tw)

    tw = [tw for tw in clean_tw if tw not in stop_words]
    lem_tw = [lemmatizer.lemmatize(i) for i in tw]
    joined_stem_tweet = ' '.join([stemmer.stem(i) for i in lem_tw])

    return joined_stem_tweet

def clean_dataset(dataset):
    x = 1 
    for idx, row in dataset.iterrows():
        print('[+] Processing Text...')
        row['Tweet'] = preprocess_tweet(str(row['Tweet']))
        print(f' Progress: {(x / len(dataset)) * 100}%')
        x += 1
    print('[i] Dataset Cleaned')

    return dataset

def convert_classifiers(dataset, adjusted_sentiments):

    counter = {}
    counter[adjusted_sentiments[0]] = 0
    counter[adjusted_sentiments[1]] = 0
    counter[adjusted_sentiments[2]] = 0
    counter[adjusted_sentiments[3]] = 0
    counter[adjusted_sentiments[4]] = 0
    counter[adjusted_sentiments[5]] = 0

    x = 1

    for i, row in dataset.iterrows():

        if (row['Label'] == 'positivity' or row['Label'] == 'positivevibes' 
            or row['Label'] == 'positivethinking' or row['Label'] == 'positivequotes'
            or row['Label'] == 'loveyourself' or row['Label'] == 'happy'
            or row['Label'] == 'love' or row['Label'] == 'motivation'):

            row['Label'] = adjusted_sentiments[0]
            print(f'Progress: {(x / len(dataset)) * 100}%')
            counter[adjusted_sentiments[0]] += 1
            x += 1

        if (row['Label'] == 'depression' or row['Label'] == 'depressionnap'
            or row['Label'] == 'depressionanxiety' or row['Label'] == 'anxiety'
            or row['Label'] == 'anxietydepression' or row['Label'] == 'sad'
            or row['Label'] == 'verysad'):

            row['Label'] = adjusted_sentiments[1]
            print(f'Progress: {(x / len(dataset)) * 100}%')
            counter[adjusted_sentiments[1]] += 1
            x += 1

        if (row['Label'] == 'afraid' or row['Label'] == 'fear'
            or row['Label'] == 'death' or row['Label'] == 'scared'
            or row['Label'] == 'scarystories'):

            row['Label'] = adjusted_sentiments[2]
            print(f'Progress: {(x / len(dataset)) * 100}%')
            counter[adjusted_sentiments[2]] += 1
            x += 1

        if (row['Label'] == 'anger' or row['Label'] == 'rage'
            or row['Label'] == 'disgusted' or row['Label'] == 'mad'
            or row['Label'] == 'hate' or row['Label'] == 'shitty' 
            or row['Label'] == 'angry' or row['Label'] == 'ridiculous'):

            row['Label'] = adjusted_sentiments[3]
            print(f'Progress: {(x / len(dataset)) * 100}%')
            counter[adjusted_sentiments[3]] += 1
            x += 1

        if (row['Label'] == 'hope' or row['Label'] == 'hopeful'
            or row['Label'] == 'wishfulthinking' or row['Label'] == 'wishing'
            or row['Label'] == 'joy' or row['Label'] == 'excited' or row['Label'] == 'kindness'):

            row['Label'] = adjusted_sentiments[4]
            print(f'Progress: {(x / len(dataset)) * 100}%')
            counter[adjusted_sentiments[4]] += 1
            x += 1

        if (row['Label'] == 'calm' or row['Label'] == 'reasonable' or row['Label'] == 'peace'
            or row['Label'] == 'peaceful' or row['Label'] == 'tranquility' 
            or row['Label'] == 'zen' or row['Label'] == 'meditation' or row['Label'] == 'blessed'):

            row['Label'] = adjusted_sentiments[5]
            print(f'Progress: {(x / len(dataset)) * 100}%')
            counter[adjusted_sentiments[5]] += 1
            x += 1

    print('[i] Labels Consolidated')
    print(f'[i] Label Counts: \n {counter}')
    return dataset


def label_to_int(dataset):

    sentiments = list(dataset['Label'])

    label_int = {}
    count = 0
    print('[+] Build Encoded Labels Table')
    for label in list(OrderedDict.fromkeys(sentiments)):
        label_int[label] = count
        count += 1

    dataset = dataset.copy()
    dataset['Class'] = None
    print('[+] Transforming Labels to Int Classes')
    x = 1

    for idx, row in dataset.iterrows():

        row['Class'] = label_int.get(row['Label'])
        print('[+] Label to Int Transformed...')
        print(f'Progress: {(x / len(dataset)) * 100}%')
        x += 1

    print('[i] Dataset Organized')
    return dataset


def gen_train_validate_test_split(dataset):

    print('[+] Splitting Dataset into Train | Validate | Test...')
    train, valid, test = np.split(dataset.sample(frac=1), [int(.6*len(dataset)), int(.8*len(dataset))])
    print('[i] Split Complete...')
    
    return train, valid, test


def save_dataset(dataset, fname):

    dataset.to_csv(f'data/{fname}.csv')
    print(f'[i] Processed, Formatted and Saved {fname}.csv')
    
    return None


class Data_Collect:

    def __init__(self, dataset_dir, classes):

        self.dataset = combine_datasets(dataset_dir, classes)

    def retrieve(self, adjusted_sentiments):

       data = fetch_valid_tweets(self.dataset)
       data = clean_dataset(data)
       data = convert_classifiers(data, adjusted_sentiments)
       data = label_to_int(data)

       train, valid, test = gen_train_validate_test_split(data)

       save_dataset(train, 'train')
       save_dataset(valid, 'valid')
       save_dataset(test, 'test')

       return None











