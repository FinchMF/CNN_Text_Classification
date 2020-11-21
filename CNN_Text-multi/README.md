# Multi-Emotion Classification
## How to Train:
        pip install -r requirements.txt
        bash exe.sh
## Additional Requirements:
In order to run this program seamlessly, you will need Twitter API keys. If you are unfamiliar with Twitter's API, please refer to my Twitter Pipeline repo to learn more: [link to repo](https://github.com/FinchMF/Twitter_Data_Pipeline)

Once you have aquired the necessary keys, add the following file in the **classifier** directory:
* config.py 

Contents of config.py:

            consumer_key = <your consumer_key>
            consumer_secret = <your consumer_secret_key>
            access_token = <your access_token>
            access_secret_token = <your access_secret_token>

## How to use trained CNN
In build.py you find the function **predict_sentiment** - this function passes:
* trained model = file: CNN_Text-Model.h5 (found in model directory)
* text_vocab_data = file: TEXT.Field (found in model directory)
* text to be classified

Load to the two variables:

 FILES:
* trained model via torch.load()
* text_vocab_data via dill.load()

THEN implement **gen_test** function.

Examples impemented beneath - **name** - function at EOF'