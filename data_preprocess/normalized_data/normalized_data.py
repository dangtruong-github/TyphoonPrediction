import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import json
import sys

# Get the directory of the current Python script (a.py)
config_directory = os.path.abspath(__file__)
current_directory = os.path.dirname(os.path.abspath(__file__))

for i in range(3):
    config_directory = os.path.dirname(config_directory)

print(current_directory)

with open(os.path.join(config_directory, 'config.json'), 'r') as f:
    config = json.load(f)

data_total_file_name = config["data_preprocess"]["aggregate_data"]["file_name"]
file_name = config["data_preprocess"]["normalized_data"]["file_name"]
year = config["data_preprocess"]["normalized_data"]["year"]
time_step = config["data_preprocess"]["normalized_data"]["time_step"]

def get_working_dir():
    current_path = Path(current_directory)
    parent_path = current_path

    for i in range(2):
        parent_path = parent_path.parent.absolute()
        
    return str(parent_path.absolute())

def normalized(data_np, mean, std, img_path=None, cols=5, bin_num=20):
    data_norm = (data_np - mean) / std

    if type(img_path) != type(None):
        array_4d_transposed = np.transpose(data_norm, (1, 0, 2, 3))  # Transpose shape: (4, 3, 5, 6)

        # Reshape the transposed array to 2D
        array_2d_k = array_4d_transposed.reshape(array_4d_transposed.shape[0], -1)

        rows = (array_2d_k.shape[0] // cols) + 1

        print(rows)
        
        
        """
        for i in range(array_2d_k.shape[0]):
            plt.hist(array_2d_k[i], bins=np.arange(array_2d_k[i].min(), array_2d_k[i].max() + bin_width, bin_width))
        print(array_2d_k.shape)
        plt.xlabel('Values')
        plt.ylabel('Frequency')        
        plt.title('Histogram')
        plt.savefig('{}.png'.format(img_path))

        gc.collect()
        

        """
        
        # Create subplots for each row with specified distance between them
        fig, axs = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), sharex=False, sharey=False, gridspec_kw={'hspace': 0.2})  # Adjust the vertical spacing here

        # Plot histogram for each row
        for i in range(rows):
            for j in range(cols):
                index_to_share = i * cols + j
                if index_to_share >= array_2d_k.shape[0]:
                    break

                min_val = array_2d_k[index_to_share].min()
                max_val = array_2d_k[index_to_share].max()
                
                bin_width = (max_val - min_val) / float(bin_num)

                axs[i][j].hist(array_2d_k[index_to_share], bins=np.arange(min_val, max_val + bin_width, bin_width))  # Set the bin width to 0.2
                axs[i][j].set_xlabel('Values')
                axs[i][j].set_ylabel('Frequency')
                axs[i][j].set_title(f'Item {index_to_share}')
                axs[i][j].grid(True)

                gc.collect()
            print(f"Finish row {i+1}")
        #plt.tight_layout()

        # Save the plot
        plt.savefig('{}.png'.format(img_path))

        gc.collect()
        

    return data_norm

def normalized_data(save_data=False):
    working_dir = get_working_dir()

    data_path = os.path.join(working_dir, "data_total", "{}.npy".format(data_total_file_name))
    
    df_path = os.path.join(working_dir, "data_total", "{}.csv".format(data_total_file_name))

    data_npy = np.load(data_path)
    df_orig = pd.read_csv(df_path)
    normalized_index = df_orig.loc[(df_orig["year"].isin(year)) & (df_orig["file_location"].str.contains("|".join(time_step)))]["index_np"].values

    print(normalized_index)
    ### normalized ###
    data_new = data_npy[normalized_index]

    np_mean = np.mean(data_new, axis=(0, 2, 3))
    np_std = np.std(data_new, axis=(0, 2, 3))

    np_mean = np.expand_dims(np_mean, axis=(0, 2, 3))
    np_std = np.expand_dims(np_std, axis=(0, 2, 3))

    #np_norm = np.random.normal(loc=5.0, scale=7.0, size=(10, 100, 10, 10))

    #np_mean_norm = np.mean(np_norm, axis=(0, 2, 3))
    #np_std_norm = np.std(np_norm, axis=(0, 2, 3))

    #np_mean_norm = np.expand_dims(np_mean_norm, axis=(0, 2, 3))
    #np_std_norm = np.expand_dims(np_std_norm, axis=(0, 2, 3))

    #normalized(data_npy, np_mean, np_std, "todas_data")
    
    #normalized(data_new, np_mean, np_std, "solo_normal_data")
    
    #normalized(np.random.normal(loc=5.0, scale=7.0, size=(10, 100, 10, 10)), 5.0, 7.0, "pure_normal_data")

    #normalized(np.random.normal(loc=0.0, scale=1.0, size=(10, 100, 10, 10)), 0.0, 1.0, "unchanged_data")

    #normalized(np_norm, np_mean_norm, np_std_norm, "unchanged_data_with_norm")

    data_normalized = normalized(data_npy, np_mean, np_std, "unnormed_filter")
    
    data_npy = None
    gc.collect()

    if save_data:
        df_to_save = pd.read_csv(df_path)
        df_to_save.to_csv(os.path.join(working_dir, "data_total", "{}.csv".format(file_name)))
        np.save(os.path.join(working_dir, "data_total", "{}.npy".format(file_name)), data_normalized)


if __name__ == "__main__":
    parent_dir = os.path.dirname(__file__)
    sys.stdout = open(os.path.join(parent_dir, 'output.txt'), 'w')
    sys.stderr = open(os.path.join(parent_dir, 'error.txt'), 'w')
    normalized_data(save_data=True)
    
    sys.stdout.close()
    sys.stderr.close()