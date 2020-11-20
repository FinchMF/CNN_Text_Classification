#!/bin/bash

if [ -d 'crawler_tweets' ];

then
    python train_CNN.py

else
    bash setup.sh
    python fetch_data.py
    mv *.csv crawler_tweets
    python train_CNN.py

fi