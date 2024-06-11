#### IMPORT NECESSARY LIBRARIES + HYPERPARAMETERS ####
from evaluation import *
from loader import *
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys

from sklearn.metrics import confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(current_directory)

with open(os.path.join(current_directory, 'config.json'), 'r') as f:
    config = json.load(f)

test_file_train = config["train"]["test"]
file_name = config["train"]["file_test_name"] if test_file_train else config["data_preprocess"]["normalized_data"]["file_name"]

batch_size = config["train"]["batch_size"]
learning_rate = config["train"]["learning_rate"]
weight_decay = config["train"]["weight_decay"]
num_epochs = config["train"]["num_epochs"]
save_path = config["train"]["save_path"]
threshold = config["train"]["threshold"]
model_type = config["train"]["model_type"]

if model_type not in ["ResNet", "CNN2D"]:
    raise Exception(f"Model type {model_type} doesn't exist")


#### RETRIEVE PATHS ####
# Data path
parent_path = current_directory

parent_data_path = os.path.join(parent_path, "data_total")

print(parent_data_path)

# Model save path
parent_model_save_path = os.path.join(parent_path, "model_save")

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

def save_model(model, optimizer, epoch, path):

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load(model, optimizer, path):

    checkpoint = torch.load(path, map_location=torch.device(device))

    print(type(checkpoint["model_state_dict"]))

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]

    return model, optimizer, epoch

#### TRAIN MODEL ####

def train(train_loader, val_loader, num_epochs, batch_print=40, id=0, time_step_used_now=0, label_now=0, weight_now=[1, 1]):
    auc_roc_score_list = []

    train_acc_list = []
    train_loss_list = []
    train_precision_neg_list = []
    train_recall_neg_list = []
    train_f1_neg_list = []
    train_precision_pos_list = []
    train_recall_pos_list = []
    train_f1_pos_list = []

    val_acc_list = []
    val_loss_list = []
    val_precision_neg_list = []
    val_recall_neg_list = []
    val_f1_neg_list = []
    val_precision_pos_list = []
    val_recall_pos_list = []
    val_f1_pos_list = []

    cur_epoch = -1

    model, criterion, optimizer = init_model(weight=weight_now)

    MODEL_SAVE_PATH = os.path.join(parent_model_save_path, "model", "{}_{}_id_{}.pt".format(model_type, save_path, id))
    JSON_SAVE_PATH = os.path.join(parent_model_save_path, "others", "{}_{}_id_{}.json".format(model_type, save_path, id))
    #NUMPY_SAVE_PATH = os.path.join(parent_folder, "standardized" if standardized else "normalized", "resnet_val_day_sensitive_{}_time_step.pkl".format(time_step))


    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(JSON_SAVE_PATH):
        with open(JSON_SAVE_PATH, "r") as file:
            f = json.load(file)
            print(f)
            auc_roc_score_list = f["auc_roc_score_list"]
            
            train_acc_list = f["train_acc_list"]
            train_loss_list= f["train_loss_list"]
            train_precision_neg_list = f["train_precision_neg_list"]
            train_recall_neg_list = f["train_recall_neg_list"]
            train_f1_neg_list = f["train_f1_neg_list"]
            train_precision_pos_list = f["train_precision_pos_list"]
            train_recall_pos_list = f["train_recall_pos_list"]
            train_f1_pos_list = f["train_f1_pos_list"]

            val_acc_list = f["val_acc_list"]
            val_loss_list= f["val_loss_list"]
            val_precision_neg_list = f["val_precision_neg_list"]
            val_recall_neg_list = f["val_recall_neg_list"]
            val_f1_neg_list = f["val_f1_neg_list"]
            val_precision_pos_list = f["val_precision_pos_list"]
            val_recall_pos_list = f["val_recall_pos_list"]
            val_f1_pos_list = f["val_f1_pos_list"]

            cur_epoch = f["epoch"]


        model, optimizer, cur_epoch = load(model, optimizer, path=MODEL_SAVE_PATH)

        #with open(NUMPY_SAVE_PATH, 'rb') as f:
        #    numpy_final_result = pickle.load(f)

        ### LOAD MODEL ###


    for epoch in range(num_epochs):
        if cur_epoch >= epoch:
            continue

        correct_samples = 0
        total_samples = 0

        loss_epoch = 0

        train_cm = [[0, 0], [0, 0]]

        print("----------------------------------------")

        model.train()

        for batch_idx, (data, label) in enumerate(train_loader):
            # Data to CUDA if possible
            data = data.to(device=device)
            label = label.to(device=device)
            label = label.to(torch.float32)

            optimizer.zero_grad()
            try:
                prob = model(data)
            except Exception as e:
                print(data.shape)
                print(e)

            pred = (nn.Sigmoid()(prob) >= threshold).squeeze()

            #print("Prob shape: {}".format(prob.shape))
            #print("Pred shape: {}".format(pred.shape))
            #print("Label shape: {}".format(label.shape))

            correct_samples += (pred == label).sum()
            total_samples += label.shape[0]

            #print((pred == label).sum())
            #print(label.shape[0])
            #print(pred)
            #print(label)
            #print("xxxxxxxxxxx")

            loss = criterion(prob, label.unsqueeze(-1))
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

            conf_mat = confusion_matrix(label.cpu().numpy(), pred.cpu().numpy(), labels=[0,1]).ravel()
            tn, fp, fn, tp = conf_mat

            train_cm[0][0] += tp
            train_cm[0][1] += tn
            train_cm[1][0] += fp
            train_cm[1][1] += fn


            if batch_idx % batch_print == batch_print - 1:
                print(f"Batch {batch_idx + 1}: Accuracy: {float(correct_samples) / float(total_samples)}")

        # Validation

        # Test
        val_acc, val_loss, val_cm = summary(val_loader, model, criterion, threshold=threshold)

        auc_roc_score = AUCROCScore(model, val_loader, save_fig=False, save_path="./sample.png")
        #final_result = test_on_1_to_21(val_val_loader, model, criterion)

        #for i in range(20):
        #    numpy_final_result[i].extend(final_result[i])
        #    print(f"Prob for {i + 1}: min {np.min(numpy_final_result[i])}, max: {np.max(numpy_final_result[i])}")

        auc_roc_score_list.append(auc_roc_score)

        train_acc_list.append(float(correct_samples) / float(total_samples + 1e-12) * 100)
        train_loss_list.append(float(loss_epoch) / float(len(train_loader)))

        train_precision_neg, train_recall_neg, train_f1_neg = f1_score(train_cm, label=0)
        train_precision_neg_list.append(train_precision_neg)
        train_recall_neg_list.append(train_recall_neg)
        train_f1_neg_list.append(train_f1_neg)

        train_precision_pos, train_recall_pos, train_f1_pos = f1_score(train_cm, label=1)
        train_precision_pos_list.append(train_precision_pos)
        train_recall_pos_list.append(train_recall_pos)
        train_f1_pos_list.append(train_f1_pos)

        val_acc_list.append(val_acc)
        val_loss_list.append(float(val_loss) / float(len(val_loader)))

        val_precision_neg, val_recall_neg, val_f1_neg = f1_score(val_cm, label=0)
        val_precision_neg_list.append(val_precision_neg)
        val_recall_neg_list.append(val_recall_neg)
        val_f1_neg_list.append(val_f1_neg)

        val_precision_pos, val_recall_pos, val_f1_pos = f1_score(val_cm, label=1)
        val_precision_pos_list.append(val_precision_pos)
        val_recall_pos_list.append(val_recall_pos)
        val_f1_pos_list.append(val_f1_pos)

        if epoch % 1 == 0:
            save_model(model=model, optimizer=optimizer, epoch=epoch, path=MODEL_SAVE_PATH)

            save_result = {
                "auc_roc_score_list": auc_roc_score_list,

                "train_acc_list": train_acc_list,
                "train_loss_list": train_loss_list,
                "train_precision_neg_list": train_precision_neg_list,
                "train_recall_neg_list": train_recall_neg_list,
                "train_f1_neg_list": train_f1_neg_list,
                "train_precision_pos_list": train_precision_pos_list,
                "train_recall_pos_list": train_recall_pos_list,
                "train_f1_pos_list": train_f1_pos_list,

                "val_acc_list": val_acc_list,
                "val_loss_list": val_loss_list,
                "val_precision_neg_list": val_precision_neg_list,
                "val_recall_neg_list": val_recall_neg_list,
                "val_f1_neg_list": val_f1_neg_list,
                "val_precision_pos_list": val_precision_pos_list,
                "val_recall_pos_list": val_recall_pos_list,
                "val_f1_pos_list": val_f1_pos_list,
                "time_step_used": time_step_used_now,
                "label": label_now,

                "epoch": int(epoch)
            }

            with open(JSON_SAVE_PATH, "w") as file:
                json.dump(save_result, file)

           # with open(NUMPY_SAVE_PATH, 'wb') as f:
           #     pickle.dump(numpy_final_result, f)

        cur_epoch = epoch

        print(f"Epoch {epoch + 1}:")

        print(f"AUC ROC Score of val list: {auc_roc_score_list[-1]}")

        print(f"Train accuracy: {train_acc_list[-1]}%")
        print(f"Train loss: {train_loss_list[-1]}")

        print(f"Val accuracy: {val_acc_list[-1]}%")
        print(f"Val loss: {val_loss_list[-1]}")

    return os.path.basename(os.path.normpath(MODEL_SAVE_PATH))


#### DATASET ####
def retrieve_time_step_and_labels():
    time_step_used = []
    label_used = []
    name_used = []

    for item in dataset_json:
        time_step_used.append(item.time_step)    
        label_used.append(item.label)
        name_used.append(item.name)

    return time_step_used, label_used, name_used

time_step_used, label_used, name_used = retrieve_time_step_and_labels()
print(time_step_used)
print(label_used)

if __name__ == "__main__":
    parent_dir = os.path.dirname(__file__)
    sys.stdout = open(os.path.join(parent_dir, 'output_train.txt'), 'w')
    sys.stderr = open(os.path.join(parent_dir, 'error_train.txt'), 'w')

    model_path_list = []

    print("Running file train")

    for i in range(len(time_step_used)):
        print(f"Currently training: {time_step_used[i]}")
        print(parent_data_path)
        print(file_name)

        train_set = CustomDataset(parent_path=parent_data_path, file_path=file_name, loc=time_step_used[i], label=label_used[i], type_dataset="train")
        val_set = CustomDataset(parent_path=parent_data_path, file_path=file_name, loc=time_step_used[i], label=label_used[i], type_dataset="val")

        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, drop_last=True)

        weight_now = [train_set.rate()]

        model_path = train(train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs, batch_print=50, id=name_used[i], time_step_used_now=time_step_used[i], label_now=label_used[i], weight_now=weight_now)

        model_path_list.append(model_path)

        with open(os.path.join(parent_dir, 'model_file.txt'), 'w') as file:
            for item in model_path_list:
                file.write(item + '\n')
        
    sys.stdout.close()
    sys.stderr.close()