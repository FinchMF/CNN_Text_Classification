#!/bin/bash

pip install -r requirements.txt
python -m spacy download en

if [ -d '/model' ];

then
    continue
else

    mkdir model

fi

