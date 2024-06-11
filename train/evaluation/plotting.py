import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def plotting_func(train_list, val_list, title, parent_path, file_save):
    if len(train_list) != len(val_list):
        print("Train and val list don't have the same length")
        return False
    len_both = len(train_list)

    plt.plot(np.arange(1, len_both + 1), train_list)
    plt.plot(np.arange(1, len_both + 1), val_list)
    plt.xlabel("Epoch")
    plt.title(title)

    plt.savefig(os.path.join(parent_path, file_save))
    plt.clf()

def plot_confusion_matrix(matrix, parent_path, file_save, check=False):
    categories = ['0', '1']
    new_matrix = [[matrix[1][1], matrix[0][0]], [matrix[0][1], matrix[1][0]]]
    print(f"New matrix: {new_matrix}")
    heatmap = sns.heatmap(new_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories[::-1], annot_kws={"size": 16})

    heatmap.set(xlabel='Predicted Label',
        ylabel= 'True Label',
        title= 'Confusion Matrix of dataset')

    fig = heatmap.get_figure()

    strFile = "./check.png" if check else os.path.join(parent_path, file_save)
    if os.path.isfile(strFile):
        os.remove(strFile)  
    
    fig.savefig(strFile) 

    #sns.show()

    plt.close()

if __name__ == "__main__":
    matrix =  [[68, 2275], [598, 12]]
    plot_confusion_matrix(matrix, 0, 0, check=True)