import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .basic_score import f1_score_pure_pred

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def summary(loader, model):
    num_correct = 0
    num_samples = 0
    total_loss = 0
    
    prob_total = None
    label_total = None

    model.eval()

    with torch.no_grad():
        for index, (data, label) in enumerate(loader):
            data = data.to(device=device)
            label = label.to(device=device)
            label = label.to(torch.float32)

            prob = model(data)
            prob = nn.Sigmoid()(prob)
            prob = prob.squeeze()

            if type(prob_total) == type(None):
                prob_total = prob
                label_total = label
            else:
                prob_total = torch.cat((prob_total, prob), dim=0)
                label_total = torch.cat((label_total, label), dim=0)

            num_samples += label.shape[0]

        if num_samples == 0:
            return None, None

    prob_total_return = prob_total.cpu().detach().numpy()
    label_total_return = label_total.cpu().detach().numpy()
    
    return prob_total_return, label_total_return

def summary_threshold(loader, model, threshold_list, save_path):
    prob_total, label_total = summary(loader, model)

    print(prob_total.shape)
    
    print(label_total.shape)

    print(threshold_list)

    precision_pos_list, recall_pos_list, f1_pos_list, precision_neg_list, recall_neg_list, f1_neg_list = [], [], [], [], [], []

    for threshold in threshold_list:
        pred = prob_total >= threshold
        
        precision_pos, recall_pos, f1_pos, precision_neg, recall_neg, f1_neg = f1_score_pure_pred(pred, label_total)

        precision_pos_list.append(precision_pos)
        recall_pos_list.append(recall_pos)
        f1_pos_list.append(f1_pos)

        precision_neg_list.append(precision_neg)
        recall_neg_list.append(recall_neg)
        f1_neg_list.append(f1_neg)

    data_dict = {
        "threshold": threshold_list,
        "precision-1": precision_pos_list,
        "recall-1": recall_pos_list,
        "f1-1": f1_pos_list,
        "precision-0": precision_neg_list,
        "recall-0": recall_neg_list,
        "f1-0": f1_neg_list,
    }

    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(save_path, index=False)



def AUCROCScore(model, data, save_fig=False, save_path=None):
    prob_total, label_total = summary(data, model)

    print(prob_total, label_total)

    if type(prob_total) == type(None):
        return 0

    auc_roc_score = roc_auc_score(label_total, prob_total)

    if save_fig:
        fpr, tpr, thresholds = roc_curve(label_total, prob_total)

        plt.figure()
        plt.plot(fpr, tpr, marker='.', label='ROC Curve (AUC = {:.2f})'.format(auc_roc_score))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.savefig(save_path)
        plt.clf()

    return auc_roc_score