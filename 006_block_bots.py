import time
import tweepy
import os
import pandas as pd
from tqdm import tqdm

auth = tweepy.OAuthHandler(os.environ["API_KEY"], os.environ["API_SECRET_KEY"])
auth.set_access_token(os.environ["ACCESS_TOKEN"], os.environ["ACCESS_TOKEN_SECRET"])

api = tweepy.API(auth)

df = pd.read_csv("./tmp/result.csv")
df.sort_values(by=["yhat"], ascending=False, inplace=True)

for username, yhat in tqdm(zip(df.username, df.yhat), desc="blocking...", total=len(df)):
    try:
        api.create_block(username)
        # time.sleep(0.1) # you may need this
    except Exception as exc:
        print(exc)
        continue
