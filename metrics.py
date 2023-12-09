from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn import metrics
from scipy import interp
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# def auc(x, y, model, curve = 'roc', file_path = 'metrics/model_roc.csv'):
#     y_pred = model.predict(x)
#     if curve == 'roc':
#         fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
#     elif curve == 'pr':
#         fpr, tpr, thresholds = metrics.precision_recall_curve(y, y_pred)
#     else:
#         raise ValueError('invalid curve type')
#     df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
#     df.to_csv(file_path, index=False)
#     return metrics.auc(fpr, tpr)

def save_roc(x, y_true, model, file = 'models/roc.csv'):
    y_pred = model.predict(x)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Calculate and plot per-position and per-nucleotide ROC curve
    for i in range(y_true.shape[1]):  # For each position in the sequence
        for j in range(y_true.shape[2]):  # For each nucleotide
            fpr, tpr, _ = roc_curve(y_true[:, i, j], y_pred[:, i, j])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            tprs.append(interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)

    sns.set_theme(style="darkgrid")
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

    # Save ROC curve data to file
    df = pd.DataFrame({
        'fpr': mean_fpr,
        'tpr': mean_tpr,
        'auc': mean_auc
    })
    df.to_csv(file, index=False)

def graph_roc_curves(files):
    sns.set_theme(style="darkgrid")
    for file in files:
        df = pd.read_csv(file)
        plt.plot(df['fpr'], df['tpr'], label=file.split('/')[-1].split('.')[0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    os.system('clear')

    graph_roc_curves([
        'metrics/model_metrics.csv'
    ])