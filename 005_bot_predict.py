import pickle
import pandas as pd
import glob


with open("./tmp/model.pkl", "rb") as fp:
    clf = pickle.load(fp)

df = pd.concat([pd.read_csv(filename) for filename in glob.glob("./tmp/agged_local_*_files.csv")])
df["compression_rate"] = df["bz2_size"] / df["original_size"]
df["inv_compression_rate"] = df["compression_rate"].apply(lambda x: abs(1-x))
df["inv_uniq_text_rate"] = df["uniq_text_rate"].apply(lambda x: abs(1-x))

usernames, X = df.username, df[["is_near_rate", "inv_compression_rate", "inv_uniq_text_rate"]]

yhats = clf.predict(X)

objs = []
for username, yhat in zip(usernames, yhats):
    judge = int(yhat>0.005)
    if judge == 1:
        objs.append({"username": username, "yhat": yhat})

pd.DataFrame(objs).to_csv("./tmp/result.csv", index=None)
