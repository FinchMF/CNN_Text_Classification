
import tw_utils as tw
import transform_data as td 
import params as p 

def fetch():

    print('[+] Requesting Data from Twitter...')
    crawler = tw.Twitter_cli()
    crawler.sentiment_crawler(p.configure()['sentiments'])
    print('[i] Tweets Collected...')

    return None

def collect():

    print('[+] Transforming Data...')
    td.Data_Collect(p.configure()['dataset_dir'], 
                    p.configure()['sentiments']).retrieve(p.configure()['sentiment_adjusted'])

    return None


def fetch_and_collect():

    fetch()
    collect()

    return None


if __name__ == '__main__':

    fetch_and_collect()
    collect()

