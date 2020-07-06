import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
df = pd.read_csv("./tmp/agged_local_files.csv")
df["compression_rate"] = df["bz2_size"] / df["original_size"]
df = df[df["compression_rate"] <= 1]

df["is_bot"] = df.username.apply(lambda x:1 if re.search(r"_bot$", x) else 0)
df.sort_values(by=["is_bot"], inplace=True)
print(df)

plt.subplots(figsize=(10,9))
ax = sns.scatterplot(x="is_near_rate", y="compression_rate", hue="is_bot", data=df, s=5)
ax.get_figure().savefig("tmp/tmp.png")
