from sklearn.metrics import roc_auc_score
import pandas as pd
import re
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("./tmp/agged_local_files.csv")

df["is_bot"] = df.username.apply(lambda x:1 if re.search(r"_bot$", x) else 0)
df["compression_rate"] = df["bz2_size"] / df["original_size"]
df = df[df["compression_rate"] <= 1]
df["inv_compression_rate"] = df["compression_rate"].apply(lambda x: abs(1-x))

print(f"only compresssion_rate auc = {roc_auc_score(df['is_bot'], df['inv_compression_rate'])}")

# is_near_rateは10分の倍率のツイートの占有率
print(f"only is_near_rate auc = {roc_auc_score(df['is_bot'], df['is_near_rate'])}")


X,y = df[["is_near_rate", "inv_compression_rate"]], df["is_bot"]

# model = SGDClassifier(loss="log", penalty="l1", tol=1/(10**3), max_iter=1000000)
model = LogisticRegression()
model.fit(X.values,y.values)

y_pred = model.predict(X.values)
print(f"logistic regression auc = {roc_auc_score(df['is_bot'], y_pred)}")


