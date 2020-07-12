import glob
import gzip
import json
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import psutil
import sys
from loguru import logger
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

def load(name):
    with gzip.open(f"tmp/user_feats/{name}", "rt") as fp:
        try:
            obj = json.load(fp)
        except json.decoder.JSONDecodeError:
            return None
    return obj

for prefix in list("0123456789"): #list("abcdefghijklmnopqrstuvwxyz_"):
    filenames = [fn.split("/")[-1] for fn in glob.glob(f"tmp/user_feats/{prefix}*")]
    logger.info(f"finish load tmp/user_feats/{prefix}*")
    objs = []
    with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as exe:
        for obj in tqdm(exe.map(load, filenames), desc="agging...", total=len(filenames)):
            if obj is None:
                continue
            objs.append(obj)

    df = pd.DataFrame(objs)
    df.to_csv(f"tmp/agged_local_{prefix}_files.csv", index=None)
