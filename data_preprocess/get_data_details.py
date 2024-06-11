import os
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import sys
import json

config_directory = os.path.abspath(__file__)

for i in range(2):
    config_directory = os.path.dirname(config_directory)

with open(os.path.join(config_directory, 'config.json'), 'r') as f:
    config = json.load(f)

data_total_file_name = config["data_preprocess"]["nc_to_npy"]["data_total_file_name"]

# Get the directory of the current Python script (a.py)
current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory)

def get_data_path():
    path = Path(current_directory)

    parent_path = path

    for _ in range(4):
        parent_path = parent_path.parent.absolute()

    data_parent_path = os.path.join(str(parent_path.absolute()), "tqluu/BigRed200/@PUBLIC/RELEASE/20240305/ncep-fnl")

    print(data_parent_path)

    return data_parent_path

def files_to_df():
    data_parent_path = get_data_path()

    data_dict = []

    current_path = Path(current_directory)
    parent_path = current_path.parent.absolute()
    path_df_loc = os.path.join(str(parent_path.absolute()), "data_preprocess/label_no_noise.csv")

    df_loc = pd.read_csv(path_df_loc)

    total_files = len(os.listdir(os.path.join(data_parent_path, "POSITIVE"))) + len(os.listdir(os.path.join(data_parent_path, "PastDomain")))

    print(f"Total files available: {total_files}")

    for index, file in enumerate(os.listdir(os.path.join(data_parent_path, "POSITIVE"))):
        if index % 1000 == 0:
            print(f"finish positive {index}")
        df_check = df_loc.loc[df_loc["file_location"].str.contains(file)]

        if df_check.empty:
            continue

        data_dict.append({
            "file_location": os.path.join("POSITIVE", file),
            "label": 1,
            "year": int(file.split("_")[1][:4]),
            "domain": "POSITIVE"
        })

    print("finish POSITIVE")

    for index, file in enumerate(os.listdir(os.path.join(data_parent_path, "PastDomain"))):
        if index % 1000 == 0:
            print(f"finish positive {index}")
        df_check = df_loc.loc[df_loc["file_location"].str.contains(file)]

        if df_check.empty:
            continue
        
        data_dict.append({
            "file_location": os.path.join("PastDomain", file),
            "label": 0,
            "year": int(file.split("_")[1][:4]),
            "domain": "PastDomain"
        })

    print("finish PastDomain")

    data_df = pd.DataFrame.from_dict(data_dict)
    path_to_save = os.path.join(str(parent_path.absolute()), "data_total", "{}.csv".format(data_total_file_name))
    data_df.to_csv(path_to_save, index=False)

if __name__ == "__main__":
    parent_dir = os.path.dirname(__file__)
    sys.stdout = open(os.path.join(parent_dir, 'output.txt'), 'w')
    sys.stderr = open(os.path.join(parent_dir, 'error.txt'), 'w')
    files_to_df()
    sys.stdout.close()
    sys.stderr.close()