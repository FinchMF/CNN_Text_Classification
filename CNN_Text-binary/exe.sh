#!/bin/bash



if [ -d 'classifier/.data/' ];

then 
    cd classifier
    python train_CNN.py
else
    bash setup.sh
    cd classifier
    python train_CNN.py

fi