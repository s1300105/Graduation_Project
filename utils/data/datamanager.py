import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import glob
from ..functions import parse

def read(path, json_file):
    fp = str(Path(path) / json_file)

    # まずは「JSON Lines 前提 + pythonエンジン」でトライ（巨大整数OK）
    try:
        return pd.read_json(fp, lines=True, engine="python")
    except ValueError:
        # JSON配列（[ {...}, {...} ]) だった場合はこちら
        try:
            return pd.read_json(fp, lines=False, engine="python")
        except ValueError:
            # それでもダメなら行ごとに標準jsonで読み込む（最強フォールバック）
            rows = []
            with open(fp, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        # 行末カンマや壊れた行をスキップ（必要に応じてログ）
                        continue
            return pd.DataFrame(rows)

def write(data_frame: pd.DataFrame, path, file_name):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)  # ディレクトリが無ければ作成

    out_path = p / file_name              # 安全なパス結合
    data_frame.to_pickle(out_path)
    print(f"[INFO] Saved: {out_path}")

def apply_filter(data_frame: pd.DataFrame, filter_func):
    return filter_func(data_frame)

def clean(data_frame: pd.DataFrame):
    return data_frame.drop_duplicates(subset="func", keep=False)

def drop(data_frame: pd.DataFrame, keys):
    for key in keys:
        del data_frame[key]

def slice_frame(data_frame: pd.DataFrame, size:int):
    data_frame_size = len(data_frame)
    return data_frame.groupby(np.arange(data_frame_size) // size)

def to_files(data_frame: pd.DataFrame, out_path):
    os.makedirs(out_path, exist_ok=True)

    for idx, row in data_frame.iterrows():
        file_name = f"{idx}.c"
        with open(out_path + file_name, 'w') as f:
            f.write(row.func)

def create_with_index(data_frame, columns):
    """
    data_frame: pandas.DataFrame もしくは (records の) iterable
    columns   : 例 ["Index", "cpg"]
    戻り値    : Index 列を index に設定した DataFrame
    """
    # DataFrame でなければ生成
    if not isinstance(data_frame, pd.DataFrame):
        data_frame = pd.DataFrame(data_frame, columns=columns)

    # 必要な列だけに絞る（存在すれば）
    cols = [c for c in columns if c in data_frame.columns]
    if cols:
        data_frame = data_frame.loc[:, cols].copy()

    # Index 列を整数化して index に設定
    if "Index" in data_frame.columns:
        data_frame["Index"] = pd.to_numeric(data_frame["Index"], errors="coerce").astype("Int64")
        data_frame = data_frame.dropna(subset=["Index"])
        data_frame["Index"] = data_frame["Index"].astype(int)
        # drop=False にしておくと列も残る（後段の参照がある場合に安全）
        data_frame.set_index("Index", drop=False, inplace=True)

    return data_frame

def inner_join_by_index(df1, df2):
    return pd.merge(df1, df2, left_index=True, right_index=True)

def get_directory_files(directory):
    return [os.path.basename(file) for file in glob.glob(f"{directory}/*.pkl")]

def get_ratio(dataset, ratio):
    approx_size = int(len(dataset) * ratio)
    return dataset[:approx_size]

def load(path, pickle_file, ratio=1):
    dataset = pd.read_pickle(path + pickle_file)
    dataset.info(memory_usage='deep')
    if ratio < 1:
        dataset = get_ratio(dataset, ratio)
    
    return dataset

def tokenize(data_frame: pd.DataFrame):
    data_frame["tokens"] = data_frame["func"].apply(parse.tokenizer)
    return data_frame[["tokens", "func"]]

