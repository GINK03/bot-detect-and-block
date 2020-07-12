import glob
from collections import Counter
import pandas as pd

"""
ユーザネームの最初の文字の頻度をチェックする
"""

c = Counter([fn.split("/")[-1][0] for fn in glob.glob("./tmp/user_feats/*")])

df = pd.DataFrame({"head": list(c.keys()), "freq": list(c.values())})

df.sort_values(by=["freq"], inplace=True)
print(df)
