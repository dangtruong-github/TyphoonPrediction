from functools import reduce
import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import os

class CustomDataset(Dataset):
    def __init__(self, parent_path, file_path, loc, label, type_dataset="train"):
        if not type_dataset in ["train", "val", "test"]:
            raise Exception(f"Type {type_dataset} not exists")
        data_path = os.path.join(parent_path, "{}.npy".format(file_path))
        df_path = os.path.join(parent_path, "{}.csv".format(file_path))

        df = pd.read_csv(df_path)
        self.data = np.load(data_path)

        loc_conditions = []

        loc_final = ["POSITIVE" if item == 0 else "_{}_".format(str(item)) for item in loc]

        #print(loc_final)
        #print(loc)
        #print("---------------------------")
        
        loc_conditions.append(df["file_location"].str.contains("|".join(loc_final)))

        if type_dataset == "train":
            loc_conditions.append(df["year"] <= 2015)
        elif type_dataset == "val":
            loc_conditions.append(df["year"] >= 2016)
            loc_conditions.append(df["year"] <= 2018)
        elif type_dataset == "test":
            loc_conditions.append(df["year"] >= 2019)
        
        if len(loc_conditions) > 0:
            loc_condition_final = reduce((lambda x, y: x & y), loc_conditions)
            df = df.loc[loc_condition_final].reset_index(drop=True)

        for i in range(len(loc_final)):
            df.loc[df["file_location"].str.contains(loc_final[i]), "label"] = int(label[i])

        self.annotations = df

    def rate(self):
        positive_samples = self.annotations["label"].sum()
        total_samples = self.__len__()
        
        return float(total_samples - positive_samples) / float(positive_samples)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        index_np = self.annotations.at[index, "index_np"]
        label = self.annotations.at[index, "label"]

        data = np.zeros((131, 17, 17))
        
        if index_np >= 0:
            data = self.data[index_np]
        
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)

        return (data, label)