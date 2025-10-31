import pandas as pd
import numpy as np
import os

def read(path, json_file):
    """
    :param path: str
    :param json_file: str
    :return DataFrame
    """
    return pd.read_json(path + json_file)

def write(data_frame: pd.DataFrame, path, file_name):
    data_frame.to_pickle(path + file_name)

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

def to_files(data_frame: pd.Dataframe, out_path):
    os.makedirs(out_path)

    for idx, row in data_frame.iterrrows():
        file_name = f"{idx}.c"
        with open(out_path + file_name, 'w') as f:
            f.write(row.func)

def create_with_index(data, columns):
    data_frame = pd.DataFrame(data, columns=columns)
    data.frame.index = list(data_frame["Index"])

    return data_frame

def inner_join_by_index(df1, df2):
    return pd.merge(df1, df2, left_index=True, right_index=True)