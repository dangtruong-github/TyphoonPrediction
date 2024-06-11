import os
from pathlib import Path
import pandas as pd
import numpy as np
import gc
import json
from aggregate_data.all_types import alpha_fe, cosine_sim, euclidean_dist
import sys

# Get the directory of the current Python script (a.py)
config_directory = os.path.abspath(__file__)
current_directory = os.path.dirname(os.path.abspath(__file__))

for i in range(3):
    config_directory = os.path.dirname(config_directory)

print(current_directory)

with open(os.path.join(config_directory, 'config.json'), 'r') as f:
    config = json.load(f)

to_agg_dict = config["data_preprocess"]["aggregate_data"]["to_agg"]
time_step = config["data_preprocess"]["aggregate_data"]["time_step"]
data_total_file_name = config["data_preprocess"]["nc_to_npy"]["file_name"]
agg_type = config["data_preprocess"]["aggregate_data"]["agg_type"]

file_name = config["data_preprocess"]["aggregate_data"]["file_name"]

to_agg = to_agg_dict.keys()

weight_func = None
if agg_type == "alpha":
    weight_func = alpha_fe
elif agg_type == "cosine":
    weight_func = cosine_sim
elif agg_type == "euclid":
    weight_func = euclidean_dist

def get_working_dir():
    current_path = Path(current_directory)
    parent_path = current_path

    for i in range(2):
        parent_path = parent_path.parent.absolute()
        
    return str(parent_path.absolute())

def load_data(file_loc, df, df_orig, data_npy, time_step):
    id_data = df.loc[df["file_location"] == file_loc]["index_np"].values[0]
    
    domain = file_loc[:-3].split("_")[0]
    string_basic = "NEGATIVE_" + file_loc[:-3].split("_")[1]
    back_step = 0
    
    if domain != "POSITIVE":
        back_step = int(file_loc[:-3].split("_")[2])

    orig = data_npy[id_data]
    currents = [orig]
    status = []

    for j in range(back_step + 1, back_step + time_step):
        string_to_search = string_basic + "_" + str(j)
        df_back_step = df_orig.loc[df_orig["file_location"].str.contains(string_to_search)]

        status.append(~df_back_step.empty)

        if df_back_step.empty:
            currents.append(np.zeros((131, 17, 17)))  
            continue

        id_back_step = df_back_step["index_np"].values[0]
        currents.append(data_npy[id_back_step])   

    
    weights = weight_func(orig, currents[1:], status)

    weights = np.expand_dims(weights, axis=(1, 2, 3))

    total_data = np.array(currents)

    concat_data_new = np.sum(total_data * weights, axis=0) 

    return concat_data_new
    
def aggregate(window=700):
    working_dir = get_working_dir()

    data_path = os.path.join(working_dir, "data_total", "{}.npy".format(data_total_file_name))

    data_npy = np.load(data_path)

    df_orig = pd.read_csv(os.path.join(working_dir, "data_total", "{}.csv".format(data_total_file_name)))

    print(data_npy.shape)
    print(len(df_orig))

    df = df_orig.loc[df_orig["file_location"].str.contains("|".join(to_agg))].reset_index(drop=True)

    aggregate_data = None
    agg_data_tmp = None
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

        ##### np_arr here #####
        np_arr = load_data(file_loc, df, df_orig, data_npy, time_step)

        np_arr = np_arr.astype("float32")

        gc.collect()

        ##### END OF EDIT #####

        np_new = np.expand_dims(np_arr, axis=0)

        if type(agg_data_tmp) == type(None):
            agg_data_tmp = np_new
            df_data_tmp = list()
        else:
            agg_data_tmp = np.concatenate((agg_data_tmp, np_new), axis=0, dtype=np.float32)

        label_key = "POSITIVE" if df.at[i, "domain"] == "POSITIVE" else "_{}_".format(file_loc[:-3].split("_")[2])
        cur_data = {
            "file_location": df.at[i, "file_location"],
            "domain": df.at[i, "domain"],
            "year": df.at[i, "year"],
            "label": to_agg_dict[label_key],
            "index_np": cur_index
        }
        
        cur_index += 1

        df_data_tmp.append(cur_data)
            
        if len(df_data_tmp) >= window:
            if os.path.exists(path_to_save):
                aggregate_data = np.load(path_to_save)
                aggregate_data = np.concatenate((aggregate_data, agg_data_tmp), axis=0, dtype=np.float32)
                
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
    pass

if __name__ == "__main__":
    parent_dir = os.path.dirname(__file__)
    sys.stdout = open(os.path.join(parent_dir, 'output.txt'), 'w')
    sys.stderr = open(os.path.join(parent_dir, 'error.txt'), 'w')
    aggregate()
    sys.stdout.close()
    sys.stderr.close()