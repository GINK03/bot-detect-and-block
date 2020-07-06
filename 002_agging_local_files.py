import glob
import gzip
import json
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


def load(filename):
    with gzip.open(filename, "rt") as fp:
        try:
            obj = json.load(fp)
        except json.decoder.JSONDecodeError:
            return None
    return obj


filenames = glob.glob("./tmp/user_feats/*")
objs = []
with ProcessPoolExecutor(max_workers=16) as exe:
    for obj in tqdm(exe.map(load, filenames), desc="agging...", total=len(filenames)):
        if obj is None:
            continue
        objs.append(obj)

df = pd.DataFrame(objs)
df.to_csv("tmp/agged_local_files.csv", index=None)
