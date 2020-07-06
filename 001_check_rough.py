import time
import copy
import asyncio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import glob
import json
import gzip
from pathlib import Path, PosixPath
import re
import datetime
import pandas as pd
import numpy as np
import random
import sys
import gzip
import os
import zlib
import bz2
from loguru import logger
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")


def get_ts_from_snowflake(status_url: str) -> datetime.datetime:
    snowflake = int(re.search("/(\d{1,})$", status_url).group(1))
    ts = datetime.datetime.fromtimestamp(((snowflake >> 22) + 1288834974657) / 1000)
    return ts


def near_1800_multiples(x):
    try:
        s = sorted([abs(x - 600 * i) for i in range(1, 10 * 24 + 1)])
        min_time = s[0]
        if min_time <= 60:
            return True
        else:
            return False
    except:
        return False


HOME = Path.home()
Path("tmp/user_feats").mkdir(exist_ok=True, parents=True)


def proc(user_dir):
    username = Path(user_dir).name
    # if not re.search("_bot$", username):
    if Path(f"tmp/user_feats/{username}").exists():
        return
    #    continue
    objs = []
    for feed in glob.glob(f"{user_dir}/FEEDS/*.gz"):
        try:
            with gzip.open(feed, "rt") as fp:
                for line in fp:
                    line = line.strip()
                    obj = json.loads(line)
                    objs.append(obj)
        except gzip.BadGzipFile:
            Path(feed).unlink()
        except EOFError:
            Path(feed).unlink()
        except zlib.error:
            Path(feed).unlink()
        except UnicodeDecodeError:
            # todo: 消してもいいかもしれない
            continue
    if len(objs) == 0:
        return
    try:
        df = pd.DataFrame(objs)
        # status_urlで重複を消去
        df.drop_duplicates(subset=["status_url"], keep="last", inplace=True)
        # usernameが入っていないことがある
        df["username"] = df["status_url"].apply(lambda x: re.search(r"https://twitter.com/(.*?)/status/", x).group(1))
        # user_dirがlowerで管理されているのでこの処理が必要
        df["username"] = df["username"].apply(lambda x: x.lower())
        df = df[df["username"] == username]
        df["ts"] = df["status_url"].apply(get_ts_from_snowflake)
        df.sort_values(by=["ts"], inplace=True)
        df["ts_shift"] = df["ts"].shift(+1)
        df["delta_ts"] = df["ts"] - df["ts_shift"]
        df["delta_seconds"] = df["delta_ts"].dt.total_seconds()

        # 平均を出して80%その差の np.sqrtの平均
        time_diversity = np.sqrt(df["delta_seconds"] - df["delta_seconds"].mean()).sort_values()[: int(len(df) * 0.8)].sum()

        # 1800で割って整数化して1800の最小の平均
        df["is_near"] = df["delta_seconds"].apply(near_1800_multiples)
        is_near_rate = df["is_near"].sum() / len(df)
        uniq_text_rate = set(df["text"].tolist()).__len__() / len(df)

        # テキストの圧縮サイズがボットのほうが小さいという仮設がある
        pid = os.getpid()
        all_text = "".join(df["text"].tolist())
        with open(f"/tmp/{pid}", "w") as fp:
            fp.write(all_text)
        original_size = Path(f"/tmp/{pid}").stat().st_size
        with bz2.open(f"/tmp/{pid}", "wt") as fp:
            fp.write(all_text)
        bz2_size = Path(f"/tmp/{pid}").stat().st_size
        obj = {"username": username, "is_near_rate": is_near_rate, "uniq_text_rate": uniq_text_rate, "original_size": original_size, "bz2_size": bz2_size}
        logger.debug(f"{obj}")
        with gzip.open(f"tmp/user_feats/{username}", "wt") as fp:
            fp.write(json.dumps(obj))
    except Exception as exc:
        tb_lineno = sys.exc_info()[2].tb_lineno
        # print(username, exc, tb_lineno, len(df))


async def async_proc(user_dir):
    proc(user_dir)


def load(dir):
    start = time.time()
    logger.info(f"start globbing {dir}...")
    dirs = glob.glob(f"{dir}/*")
    elapsed = time.time() - start
    logger.info(f"globbing {dir}, elapsed = {elapsed:0.06f}")
    return dirs


async def load_wrapper():
    files = glob.glob(f"{HOME}/nvme0n1/*")
    return [files]
    # dirs = glob.glob(f"{HOME}/.mnt/nfs/favs*")
    with ThreadPoolExecutor(max_workers=len(dirs)) as exe:
        ret = list(exe.map(load, dirs))
    # return await asyncio.gather(*[load(dir) for dir in glob.glob(f"{HOME}/.mnt/nfs/favs*")])
    return ret


user_dirs = []
for chunk in asyncio.run(load_wrapper()):
    user_dirs.extend(chunk)
random.shuffle(user_dirs)
logger.info(f'total size = {len(user_dirs)}')

if os.environ.get("NUM"):
    NUM = int(os.environ["NUM"])
else:
    NUM = 16


def wrapper(chunk):
    async def _wrapper(chunk):
        await asyncio.gather(*[asyncio.create_task(async_proc(user_dir)) for user_dir in chunk])
    asyncio.run(_wrapper(chunk))


if os.environ.get("MULTIPLE"):
    chunks = []
    for i in range(0, len(user_dirs), 5):
        chunk = user_dirs[i: i + 5]
        chunks.append(chunk)
    with ProcessPoolExecutor(max_workers=NUM) as exe:
        for _ in tqdm(exe.map(wrapper, chunks), desc="proc...", total=len(chunks)):
            _
else:

    async def wrapper(chunk):
        await asyncio.gather(*[asyncio.create_task(async_proc(user_dir)) for user_dir in chunk])

    for i in tqdm(range(0, len(user_dirs), 5)):
        chunk = user_dirs[i: i + 5]
        asyncio.run(wrapper(chunk))
