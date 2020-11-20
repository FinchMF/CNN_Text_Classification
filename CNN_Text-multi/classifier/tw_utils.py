from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import TweepError
from tweepy import parsers


import time
import json
from datetime import datetime, date, timedelta
import pandas as pd

import config

#-------# AUTHENTICATION #-------#

class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(config.consumer_key, config.consumer_secret)
        auth.set_access_token(config.access_token, config.access_token_secret)

        return auth

#-------# CLIENT #-------#

class Twitter_cli():

    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.client = API(self.auth, wait_on_rate_limit=True)
        self.twitter_user = twitter_user


    def get_twitter_client_api(self):
        return self.client

    def check_twitter_user_exists(self, username):
        try:
            self.client.get_user(username)
            return True
        except Exception:
            return False


    def search_twitter(self, from_date, num_tw, only_text=False, geocode=None, search_query=None):
        search_words = search_query
        dates = from_date

        tweets = Cursor(self.client.search,
                        q=search_words,
                        tweet_mode='extended',
                        lang='en',
                        since=dates[0],
                        until=dates[1],
                        geocode=geocode,
                        include_retweets=True).items(num_tw)

        list_of_tweets = []
        if only_text:
            for tweet in tweets:
                list_of_tweets.append(tweet.full_text)
        else:
            filtered_tweets = [[tweet.id_str, tweet.created_at, tweet.full_text, tweet.retweet_count,
                                tweet.author.screen_name, tweet.author.name, tweet.coordinates, tweet.source,
                                tweet.in_reply_to_status_id_str, tweet.in_reply_to_user_id_str, tweet.in_reply_to_screen_name,
                                tweet.retweeted, tweet.entities['hashtags']] for tweet in tweets]
   
            for t in filtered_tweets:
                keys = ['tweet_id', 'time', 'text', 'retweet_count', 'user','user_name','location', 'source','in_reply_to_status_id', 
                        'in_reply_to_status_user_id', 'in_reply_to_screen_name', 'is_retweet', 'hashtags']
                twitter_dict = dict(zip(keys, t))
                list_of_tweets.append(twitter_dict)
               

                
            for tweet in list_of_tweets:
                for k,v in tweet.items():
                    if k == 'time':
                        v = v.strftime("%m/%d/%Y, %H:%M:%S")
                        tweet['time'] = v

        return list_of_tweets



    def get_user_timeline_tweets(self, num_tw):
        tweets = []
        filtered_tweets = [[tweet.id_str, tweet.created_at, tweet.full_text, tweet.retweet_count,
                                tweet.author.screen_name, tweet.author.name, tweet.coordinates, tweet.source,
                                tweet.in_reply_to_status_id_str, tweet.in_reply_to_user_id_str, tweet.in_reply_to_screen_name,
                                tweet.retweeted, tweet.place] for tweet in Cursor(self.client.user_timeline,
                                                                     id=self.twitter_user,
                                                                     tweet_mode='extended').items(num_tw)]
                                                                   
        for t in filtered_tweets:
            keys = ['tweet_id', 'time', 'text', 'retweet', 'user','user_name','location', 'source','in_reply_to_status_id', 
                    'in_reply_to_status_user_id', 'in_reply_to_screen_name', 'is_retweet', 'place']
            twitter_dict = dict(zip(keys, t))
            tweets.append(twitter_dict)
        
        for tweet in tweets:
            for k,v in tweet.items():
                if k == 'time':
                    v = v.strftime("%m/%d/%Y, %H:%M:%S")
                    tweet['time'] = v
        
        return tweets




    def get_all_user_tweets(self, screenname, max_id=None):

        all_tweets = []

        try:
            if max_id:
                new_tweets = self.client.user_timeline(screen_name=screenname,
                                                    count=200,
                                                    max_id=max_id,
                                                    tweet_mode='extended')
            else:
                new_tweets = self.client.user_timeline(screen_name=screenname,
                                                    count=200,
                                                    tweet_mode='extended')
        except TweepError:
            print(f'limit error on {screenname} during initial request')
            print('waiting 15min...')
            time.sleep(900)
            print('waking back up...')
            print(f'now retrieving tweets from {screenname}')
            new_tweets = self.client.user_timeline(screen_name=screenname,
                                    count=200,
                                    tweet_mode='extended')
            print(f'recieved new tweets list from {screenname}')


        all_tweets.extend(new_tweets)

        oldest = all_tweets[-1].id - 1


        while len(new_tweets) > 0:
            print(f'getting tweets before {oldest}')

            try:
                new_tweets = self.client.user_timeline(screen_name=screenname,
                                                        count=200,
                                                        max_id=oldest,
                                                        tweet_mode='extended')
            except TweepError:
                print(f'limit error on {screenname} while retrieving before {oldest}')
                print('waiting 15min...')
                time.sleep(900)
                print('waking back up...')
                print(f'now retrieving more tweets from {screenname} before {oldest}')
                new_tweets = self.client.user_timeline(screen_name=screenname,
                                                        count=200,
                                                        max_id=oldest,
                                                        tweet_mode='extended')

                continue


            all_tweets.extend(new_tweets)

            oldest = all_tweets[-1].id - 1


            print(f'...{len(all_tweets)} tweets downloaded so far')


        outtweets = [[tweet.id_str, tweet.created_at, tweet.full_text, tweet.retweet_count,
                        tweet.author.screen_name, tweet.author.name, tweet.coordinates, tweet.source,
                        tweet.in_reply_to_status_id_str, tweet.in_reply_to_user_id_str, tweet.in_reply_to_screen_name,
                        tweet.retweeted, tweet.place] for tweet in all_tweets]


        organized_outtweets = []
        for t in outtweets:
            keys = ['tweet_id', 'time', 'text', 'retweet_count', 'user','user_name','location', 'source','in_reply_to_status_id', 
                    'in_reply_to_status_user_id', 'in_reply_to_screen_name', 'is_retweet', 'place']
            twitter_dict = dict(zip(keys, t))
            organized_outtweets.append(twitter_dict)

        for tweet in organized_outtweets:
            for k,v in tweet.items():
                if k == 'time':
                    v = v.strftime("%m/%d/%Y, %H:%M:%S")
                    tweet['time'] = v

        return organized_outtweets




    def sentiment_crawler(self, classes):

        for cl in classes:

            print(f'starting crawl: {cl}')

            try:
                tweets = []
                count = 0
                for tweet in Cursor(self.client.search,
                                    q=f'#{cl}',
                                    lang='en',
                                    since='2020-01-01',
                                    tweet_mode='extended',
                                    count=200).pages(500):

                    for t in tweet:
                        text = t.full_text.replace('&amp;', '&').replace(',','').replace('RT', '')
                        print(text)
                        tweets.append(text)
                        time.sleep(1e-3)
                    print(f'PAGE: {count} | CLASSIFIER: {cl}')
                    count +=1
                    
                pd.DataFrame(tweets).to_csv(f'crawler_tweets/{cl}.csv')
            
            except Exception as e:
                print(e)
                pd.DataFrame(tweets).to_csv(f'{cl}.csv')

        return None






