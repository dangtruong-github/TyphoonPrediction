import numpy as np

from sklearn.metrics import confusion_matrix

def precision_score(conf_mat, label=1):
    tp = conf_mat[0][1 - label]
    fp = conf_mat[1][1 - label] + 1e-12

    return float(tp) / float(tp + fp)

def recall_score(conf_mat, label=1):
    tp = conf_mat[0][1 - label]
    fn = conf_mat[1][label] + 1e-12

    return float(tp) / float(tp + fn)

def f1_score(conf_mat, label=1):
    precision = precision_score(conf_mat, label=label)
    recall = recall_score(conf_mat, label=label)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)

    return precision, recall, f1

def f1_score_pure_pred(pred, label):
    tn, fp, fn, tp = confusion_matrix(label, pred, labels=[0,1]).ravel()

    conf_mat = [[tp, tn], [fp, fn]]

    precision_pos, recall_pos, f1_pos = f1_score(conf_mat, label=1) 
    precision_neg, recall_neg, f1_neg = f1_score(conf_mat, label=0) 

    return precision_pos, recall_pos, f1_pos, precision_neg, recall_neg, f1_neg