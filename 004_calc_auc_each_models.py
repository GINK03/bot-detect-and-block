from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import glob
from sklearn.metrics import roc_auc_score
from concurrent.futures import ProcessPoolExecutor
import lightgbm as lgb
import seaborn as sns
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import japanize_matplotlib
plt.subplots(figsize=(10,9))

df = pd.concat([pd.read_csv(filename) for filename in glob.glob("./tmp/agged_local_*_files.csv")])

df["is_bot"] = df.username.apply(lambda x:1 if re.search(r"_bot$", x) else 0)
df["compression_rate"] = df["bz2_size"] / df["original_size"]
df = df[df["compression_rate"] <= 1]
df["inv_compression_rate"] = df["compression_rate"].apply(lambda x: abs(1-x))

df["inv_uniq_text_rate"] = df["uniq_text_rate"].apply(lambda x: abs(1-x))

print(f"only compresssion_rate auc = {roc_auc_score(df['is_bot'], df['inv_compression_rate'])}")

# is_near_rateは10分の倍率のツイートの占有率
print(f"only is_near_rate auc = {roc_auc_score(df['is_bot'], df['is_near_rate'])}")

# uniq_text_rateだけのAUC print(f"only uniq_text_rate auc = {roc_auc_score(df['is_bot'], df['inv_uniq_text_rate'])}") 
X,y = df[["is_near_rate", "inv_compression_rate", "inv_uniq_text_rate"]], df["is_bot"]

# model = SGDClassifier(loss="log", penalty="l1", tol=1/(10**3), max_iter=1000000)
model = LogisticRegression()
model.fit(X.values,y.values)

y_pred = model.predict(X.values)
print(f"logistic regression auc = {roc_auc_score(df['is_bot'], y_pred)}")


param = {
    "objective": "binary",
    "metric": "auc",
    "max_depth": 3,
    "num_leaves": 3,
    "learning_rate": 0.01,
    "bagging_fraction": 0.1,
    "feature_fraction": 0.1,
    "lambda_l1": 0.3,
    "lambda_l2": 0.3,
    "bagging_seed": 777,
    "verbosity": -1,
    "seed": 777,
}
def train(xs, ys, param, verbose, early_stopping_rounds, n_estimators):
    if isinstance(xs, (pd.DataFrame)):
        print('input xs, ys, XT may be pd.DataFrame, so change to np.array')
        xs = xs.values
        ys = ys.values
    xs_trn, xs_val, ys_trn, ys_val = train_test_split(xs, ys, test_size=0.25, shuffle=False)

    trn_data = lgb.Dataset(xs_trn, label=ys_trn, categorical_feature=[])
    val_data = lgb.Dataset(xs_val, label=ys_val, categorical_feature=[])
    num_round = n_estimators
    clf = lgb.train(param, trn_data, num_round, valid_sets=[
        trn_data, val_data], verbose_eval=verbose, early_stopping_rounds=early_stopping_rounds)
    test_predictions = clf.predict(xs_val)
    eval_loss = roc_auc_score(ys_val, clf.predict(xs_val))
    print(f'end eval_loss={eval_loss}')
    return (test_predictions, xs_val, ys_val)

test_predictions, xs_val, ys_val = train(xs=X, ys=y, param=param, verbose=1, early_stopping_rounds=5, n_estimators=1000)


def save_each_patterm(th):   
    p = test_predictions >= th
    rate = p.sum()/len(p)
    df = pd.DataFrame({"is_near_rate": xs_val[:, 0], "compression_rate": np.abs(1-xs_val[:, 1]), "inv_uniq_text_rate": xs_val[:, 2], "is_bot": p})
    ax = sns.scatterplot(x="is_near_rate", y="compression_rate", hue="is_bot", data=df, s=5)
    ax.set(title="機械的周期性 vs 圧縮サイズ/オリジナルサイズ", xlabel="機械的率周期性", ylabel="圧縮サイズ/オリジナルサイズ")
    ax.get_figure().savefig(f"tmp/tmp_th={th}_rate={rate:0.04f}.png")

ths = [0.001, 0.005, 0.01, 0.03, 0.1, 0.2]

with ProcessPoolExecutor(max_workers=16) as exe:
    exe.map(save_each_patterm, ths)
