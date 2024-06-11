import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from loader import *
from models import *
from evaluation import *

import os
import pandas as pd
import json
import sys
import gc

import numpy as np
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory)

#### RETRIEVE PATHS ####
# Data path
path = Path(current_directory)

parent_path = path

for i in range(1):
    parent_path = parent_path.parent.absolute()

data_dir = os.path.join(str(parent_path.absolute()), "data_total")
config_path = os.path.join(str(parent_path.absolute()), "config.json")

with open(config_path, "r") as file:
    config = json.load(file)

data_file_name = config["data_preprocess"]["normalized_data"]["file_name"]
batch_size = config["train"]["batch_size"]
learning_rate = config["train"]["learning_rate"]
weight_decay = config["train"]["weight_decay"]
num_epochs = config["train"]["num_epochs"]
save_path = config["train"]["save_path"]
model_index = config["train"]["model_index"]
summary_each_time_step = config["train"]["summary_each_time_step"]
summary_threshold_all = config["train"]["summary_threshold_all"]
model_type = config["train"]["model_type"]

data_path = os.path.join(data_dir, "{}.npy".format(data_file_name))
df_path = os.path.join(data_dir, "{}.csv".format(data_file_name))

print(data_dir)

print(f"Parent path: {parent_path}")
model_dir = os.path.join(str(parent_path.absolute()), "model_save", "model")
json_dir = os.path.join(str(parent_path.absolute()), "model_save", "others")

# Load data
data_np = np.load(data_path)

#### MODEL, LOSS, OPTIMIZER ####

def init_model(weight=None):
    model = None

    if model_type == "ResNet":
        model = Resnet(in_channels=131, num_class=2).to(device=device)
    elif model_type == "CNN2D":
        model = CNN2D().to(device=device)

    weight_now_fin = [1.0] if weight is None else weight
    class_weights = torch.tensor(weight_now_fin, dtype=torch.float32)
    
    #total_samples = train_set.rate()
    #total_samples[1] *= 2

    #class_weights = torch.tensor(np.sum(total_samples) / (2 * np.array(total_samples)), dtype=torch.float32)
    
    print(class_weights)

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return model, criterion, optimizer

def load(model, optimizer, path):

    checkpoint = torch.load(path, map_location=torch.device(device))

    #print(type(checkpoint["model_state_dict"]))

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]

    return model, optimizer, epoch

"""## Custom datasets"""

threshold_list = np.arange(0+0.05, 1+0.05, 0.05, dtype=np.float32)

#### EVAL ####

if __name__ == "__main__":
    parent_dir = os.path.dirname(__file__)
    sys.stdout = open(os.path.join(parent_dir, 'output_eval.txt'), 'w')
    sys.stderr = open(os.path.join(parent_dir, 'error_eval.txt'), 'w')

    model_files = []

    with open(os.path.join(current_directory, 'model_file.txt'), 'r') as file:
        model_files = file.read().split('\n')

    # If there is an empty string at the end, remove it
    if model_files[-1] == '':
        model_files.pop()

    for index, file_path in enumerate(model_files):
        #if index != model_index:
        #    continue

        #if index > model_index:
        #    break

        print(f"Start eval dataset {index}")

        model_path = os.path.join(model_dir, file_path)
        json_path = os.path.join(json_dir, "{}.json".format(file_path[:-3]))

        json_summary_file = None

        with open(json_path, "r") as file:
            json_summary_file = json.load(file)

        time_step_used_now = json_summary_file["time_step_used"]
        label_now = json_summary_file["label"]

        for i in range(0, 20):
            if not i in time_step_used_now:
                time_step_used_now.append(i)
                if i <= index:
                    label_now.append(1)
                else:
                    label_now.append(0)

        # Model summary
        test_set = CustomDataset(parent_path=data_dir, file_path=data_file_name, loc=time_step_used_now, label=label_now, type_dataset="test")

        weight_now = test_set.rate()

        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        model, criterion, optimizer = init_model(weight=weight_now)
        model, _, epoch = load(model, optimizer, model_path)
        test_acc, test_loss, cm = summary(test_loader, model, criterion)
        
        print("after test_loader_orig")

        parent_path_save_fig = os.path.join(json_dir, file_path[:-3])

        if not os.path.isdir(parent_path_save_fig):
            os.mkdir(parent_path_save_fig)

        AUCROCScore(model, test_loader, save_fig=True, save_path=os.path.join(parent_path_save_fig, "auc_roc.png"))

        print("Path to save: {}".format(os.path.join(parent_path_save_fig, "confusion_matrix.png")))
        print(f"Confusion matrix total: {cm}")
        plot_confusion_matrix(cm, parent_path_save_fig, "confusion_matrix.png")

        print(f"Dataset id: {index}")

        if summary_threshold_all:
            summary_threshold(test_loader, model, threshold_list, save_path=os.path.join(parent_path_save_fig, "summary_threshold.csv"))

        if summary_each_time_step:
            test_set = None
            test_loader = None
            gc.collect()
            dict_data = {
                "time_step": [],
                "test_acc": [],
                "test_loss": []
            }
            for j in range(len(time_step_used_now)):
                cur_time_step = [time_step_used_now[j]]
                cur_label = [label_now[j]]

                test_set_now = CustomDataset(parent_path=data_dir, file_path=data_file_name, loc=cur_time_step, label=cur_label, type_dataset="test")
                test_loader_now = DataLoader(test_set_now, batch_size=batch_size, shuffle=False)

                test_acc, test_loss, cm_now = summary(test_loader_now, model, criterion)

                print(f"t-{cur_time_step}: Acc: {test_acc}, Loss: {test_loss}, Confusion matrix: {cm_now}")

                dict_data["time_step"].append(int(cur_time_step[0]))
                dict_data["test_acc"].append(test_acc)
                dict_data["test_loss"].append(test_loss)
            
                test_set_now = None
                test_loader_now = None
                gc.collect()
            df = pd.DataFrame.from_dict(dict_data)

            df.to_csv(os.path.join(parent_path_save_fig, "time_step_analysis.csv".format(index)))    

        # Data summary 
        
        for key, item in json_summary_file.items():
            if key[:4] != "val_":
                continue
            
            title = key[4:]
            val_data_to_plot = item
            train_key_to_plot = "train_" + title
            train_data_to_plot = json_summary_file[train_key_to_plot]

            file_name_to_save = title + ".png"

            plotting_func(train_data_to_plot, val_data_to_plot, title, parent_path_save_fig, file_name_to_save)


    sys.stdout.close()
    sys.stderr.close()