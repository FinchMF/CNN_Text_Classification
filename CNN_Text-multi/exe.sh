#!/bin/bash

if [ -d 'crawler_tweets' ];

then
    python train_CNN.py

else
    bash setup.sh

    python fetch_data.py
    # incase some csv-files do not properly save in 'crawler_tweets'  
    mv *.csv crawler_tweets
    
    python train_CNN.py

fi