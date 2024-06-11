import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def summary(loader, model, criterion, threshold=0.5):
    num_correct = 0
    num_samples = 0
    total_loss = 0

    model.eval()

    acc = 0

    cm = [[0, 0], [0, 0]] # tp 00, tn 01, fp 10, fn 11

    with torch.no_grad():
        for index, (data, label) in enumerate(loader):
            data = data.to(device=device)
            label = label.to(device=device)
            label = label.to(torch.float32)

            prob = model(data)

            pred = nn.Sigmoid()(prob.squeeze()) >= threshold

            if index % 10 == 9:
                pass

            num_correct += (pred == label).sum()
            num_samples += label.shape[0]
            loss = criterion(prob, label.unsqueeze(-1))

            total_loss += loss.item()

            try:
                conf_mat = confusion_matrix(label.cpu().numpy(), pred.cpu().numpy(), labels=[0,1]).ravel()

                #print(conf_mat)

                tn, fp, fn, tp = conf_mat

                cm[0][0] += tp
                cm[0][1] += tn
                cm[1][0] += fp
                cm[1][1] += fn

            except Exception as e:
                print(e)

        if num_samples == 0:
            return 0, 0, [[0, 0], [0, 0]]
        acc = float(num_correct)/float(num_samples) * 100.0
    return acc, total_loss, cm
