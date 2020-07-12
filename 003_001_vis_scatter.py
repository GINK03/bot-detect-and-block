import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import re
import glob

"""
Dropbox参照: https://www.dropbox.com/s/sjlqmw5pk1j1zzu/bot-detect-and-block-snapshot_20200712.zip?dl=0
"""

df = pd.concat([pd.read_csv(filename) for filename in glob.glob("./tmp/agged_local_*_files.csv")])
df["compression_rate"] = df["bz2_size"] / df["original_size"]
df = df[df["compression_rate"] <= 1]
df = df.sample(n=1000000)

df["is_bot"] = df.username.apply(lambda x:1 if re.search(r"_bot$", x) else 0)
df.sort_values(by=["is_bot"], inplace=True)

plt.subplots(figsize=(10,9))
ax = sns.scatterplot(x="is_near_rate", y="compression_rate", hue="is_bot", data=df, s=5)
ax.set(title="機械的周期性 vs 圧縮サイズ/オリジナルサイズ", xlabel="機械的率周期性", ylabel="圧縮サイズ/オリジナルサイズ")

ax.get_figure().savefig("tmp/tmp.png")
