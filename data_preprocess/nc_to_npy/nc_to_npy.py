import os
from pathlib import Path
import pandas as pd
import numpy as np
import gc
import json
import sys
import h5py

# Get the directory of the current Python script (a.py)
config_directory = os.path.abspath(__file__)
current_directory = os.path.dirname(os.path.abspath(__file__))

for i in range(3):
    config_directory = os.path.dirname(config_directory)

print(current_directory)

with open(os.path.join(config_directory, 'config.json'), 'r') as f:
    config = json.load(f)

features_2d = config["data_preprocess"]["nc_to_npy"]["features_2d"]
features_1d = config["data_preprocess"]["nc_to_npy"]["features_1d"]
file_name = config["data_preprocess"]["nc_to_npy"]["file_name"]
data_total_file_name = config["data_preprocess"]["nc_to_npy"]["data_total_file_name"]

def get_working_dir():
    current_path = Path(current_directory)
    parent_path = current_path

    for i in range(2):
        parent_path = parent_path.parent.absolute()
        
    return str(parent_path.absolute())

def get_data_path():
    path = Path(current_directory)

    parent_path = path

    for _ in range(5):
        parent_path = parent_path.parent.absolute()

    data_parent_path = os.path.join(str(parent_path.absolute()), "tqluu/BigRed200/@PUBLIC/RELEASE/20240305/ncep-fnl")

    #print(data_parent_path)

    return data_parent_path

def load_data(file):
    orig_path = get_data_path()
    # Open the NetCDF file
    data_2d = ["ugrdprs", "vgrdprs", "vvelprs", "tmpprs", "hgtprs", "rhprs"]
    data_1d = ['tmptrp', 'landmask', 'pressfc', 'tmpsfc', 'hgttrp']
    
    concat_data = np.zeros((131, 17, 17))

    with h5py.File(os.path.join(orig_path, file), 'r') as f:
        # Access the dataset you want

        concat_2d_data = np.array([f[variable][:] for variable in data_2d])[:, :, :21]
        concat_2d_data = np.moveaxis(concat_2d_data, -1, 0).reshape(-1, *concat_2d_data.shape[-2:])
        #print(concat_2d_data.shape)
        concat_1d_data = np.array([f[variable][:] for variable in data_1d]).reshape(-1, *concat_2d_data.shape[-2:])
        #print(concat_1d_data.shape)

        concat_data = np.concatenate([concat_2d_data, concat_1d_data], axis=0)

        nan_data = np.isnan(concat_data).any()

        if nan_data:
            concat_data = np.nan_to_num(concat_data, nan=0.0)

            print(f"Fill {file}")

    concat_data = concat_data.astype("float32")

    gc.collect()        
    
    return concat_data

def transform_data(window=700):
    working_dir = get_working_dir()

    data_path = os.path.join(working_dir, "data_total", "{}.csv".format(data_total_file_name))

    df = pd.read_csv(data_path)

    aggregate_data = None
    agg_data_tmp = None
    df_data = list()
    df_data_tmp = list()
    cur_index = 0
    start_index = 0

    df_data_df_old = pd.DataFrame(columns=["file_location", "domain", "year", "label", "index_np"])

    path_to_save = os.path.join(working_dir, "data_total", "{}.npy".format(file_name))

    df_path_to_save = os.path.join(working_dir, "data_total", "{}.csv".format(file_name))

    if os.path.exists(df_path_to_save):
        #aggregate_data = np.load(path_to_save)
        df_data_df_old = pd.read_csv(df_path_to_save)

        #aggregate_data = aggregate_data[:df_data_df_old["index_np"].max()+1]

        cur_index = df_data_df_old["index_np"].max() + 1

        print(df_data_df_old)

        #print(aggregate_data.shape)
        print(cur_index)
        start_index = cur_index

    for i in range(start_index, len(df)):
        file_loc = df.at[i, "file_location"]

        np_arr = load_data(file_loc)

        print(np_arr[0, 0, :])

        np_new = np.expand_dims(np_arr, axis=0)

        if type(agg_data_tmp) == type(None):
            agg_data_tmp = np_new
            df_data_tmp = list()
        else:
            agg_data_tmp = np.concatenate((agg_data_tmp, np_new), axis=0, dtype=np.float32)

        cur_data = {
            "file_location": file_loc.split("/")[1],
            "domain": df.at[i, "domain"],
            "year": df.at[i, "year"],
            "label": df.at[i, "label"],
            "index_np": cur_index
        }
        
        cur_index += 1

        df_data_tmp.append(cur_data)
            
        if len(df_data_tmp) >= window:
            if os.path.exists(path_to_save):
                aggregate_data = np.load(path_to_save)
                aggregate_data = np.concatenate((aggregate_data, agg_data_tmp), axis=0)
                
                df_data_tmp_df = pd.DataFrame.from_dict(df_data_tmp)
                
                df_data_df_old = pd.concat([df_data_df_old, df_data_tmp_df])
            else:
                aggregate_data = agg_data_tmp
                
                df_data_df_old = pd.DataFrame.from_dict(df_data_tmp)
                
            np.save(path_to_save, aggregate_data)

            df_data_df_old.to_csv(df_path_to_save, index=False)
            
            print(aggregate_data.shape)
            
            aggregate_data = None
            agg_data_tmp = None
            df_data_tmp = list()
            
            gc.collect()
            print(df_data_df_old["index_np"].max())
            print("Success")

        if (i + 1) % 500 == 0:
            print(file_loc)
            print(cur_index)

    print(len(df))
            
    if os.path.exists(path_to_save):
        aggregate_data = np.load(path_to_save)
        aggregate_data = np.concatenate((aggregate_data, agg_data_tmp), axis=0)

        df_data_tmp_df = pd.DataFrame.from_dict(df_data_tmp)

        df_data_df_old = pd.concat([df_data_df_old, df_data_tmp_df])
    else:
        aggregate_data = agg_data_tmp

        df_data_df_old = pd.DataFrame.from_dict(df_data_tmp)

    np.save(path_to_save, aggregate_data)

    df_data_df_old.to_csv(df_path_to_save, index=False)
    
    aggregate_data = None
    
    gc.collect()

if __name__ == "__main__":
    parent_dir = os.path.dirname(__file__)
    sys.stdout = open(os.path.join(parent_dir, 'output.txt'), 'w')
    sys.stderr = open(os.path.join(parent_dir, 'error.txt'), 'w')
    transform_data()
    sys.stdout.close()
    sys.stderr.close()
