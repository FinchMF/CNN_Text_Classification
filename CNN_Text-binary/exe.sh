#!/bin/bash

if [ -d '.data/' ];

then 
    
    python train_CNN.py
else

    bash setup.sh
    python train_CNN.py

fi