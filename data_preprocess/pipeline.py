import os
import json

from get_data_details import files_to_df
from nc_to_npy.nc_to_npy import transform_data
from aggregate_data.aggregate_data import aggregate
from normalized_data.normalized_data import normalized_data

def get_working_dir():
    current_path = os.path.abspath(__file__)
    parent_path = current_path

    for i in range(2):
        parent_path = os.path.dirname(parent_path)
        
    return parent_path

def open_config():
    parent_path = get_working_dir()
    config_path = os.path.join(parent_path, "config.json")

    with open(config_path, "r") as file:
        config = json.load(file)

    return config

def pipeline():
    parent_path = get_working_dir()
    config = open_config()

    files_to_df()
    transform_data()
    aggregate()
    orig_data_path = os.path.join(parent_path, "data_total", config["nc_to_npy"]["file_name"])
    if os.exists(orig_data_path):
        os.remove(orig_data_path)
    normalized_data(save_data=True)
    agg_data_path = os.path.join(parent_path, "data_total", config["nc_to_npy"]["aggregate_data"])
    if os.exists(agg_data_path):
        os.remove(agg_data_path)

if __name__ == "__main__":
    
    pipeline()