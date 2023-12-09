from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn import metrics
from scipy import interp
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils import *
import preprocessing

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

def activity_test(discriminator, rna, chromosome, start, end, view_length=23, bind_site=-1, plot=True):
    
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    if bind_site == -1: bind_site = (start + end) // 2
        
    ohe_rna = np.concatenate([ohe_base(base) for base in rna], axis=0)
    seq = preprocessing.fetch_genomic_sequence(chromosome, start, end)
    ohe_seq = np.concatenate([ohe_base(base) for base in seq], axis=0)
    epigenomic_signals = preprocessing.fetch_epigenomic_signals(chromosome, start, end)
    epigenomic_seq = np.concatenate([ohe_seq, epigenomic_signals], axis=1)
    
    activity_scores = []
    for i in range(end - start - view_length):
        # if discriminator.name == 'conv_discriminator':
            activity_score = discriminator([
                np.expand_dims(epigenomic_seq[i:i+view_length], axis=0), 
                np.expand_dims(ohe_rna, axis=0)
            ])
            activity_scores.append(activity_score[0][0])
        # else:
        #     pass # finish
    activity_scores = np.array(activity_scores)
    moving_averages = moving_average(activity_scores, 23)
        
    if plot:
        x = np.arange(start + view_length, end - view_length + 1)
        plt.figure(figsize=(8, 3))
        plt.plot(x, moving_averages, label=rna)
        # plt.plot(x, activity_scores[11:-11], label='test')
        plt.axvline(x=bind_site, color='orange', linestyle='dotted', label='bind site')
        plt.title('GRNA activity over chromosome ' + chromosome)
        plt.xlabel('genomic position')
        plt.ylabel('predicted activity (0-1)')
        plt.legend()
        plt.show()
    
    return activity_scores

def perturbation_analysis(discriminator, chromosome, start, end, bind_site, rna, perturbation_length, base='N'):
    RNA = rna
    heatmap = np.zeros((len(RNA) - perturbation_length + 1,777))
    _,original = activity_test(discriminator=discriminator,rna=RNA,chromosome=chromosome,start=start,end=end,bind_site=bind_site, plot=False)
    middle = len(original)//2
    orig_target_mean = np.mean(original[middle-20:middle+50])
    perturbed_target_means = np.zeros(len(RNA) - perturbation_length + 1)
    
    for i in range(0, len(RNA)-perturbation_length + 1):
        perturbed_grna = list(RNA)
        perturbed_grna[i:i + perturbation_length] = base * perturbation_length
        perturbed_grna = ''.join(perturbed_grna)
        _, averages = activity_test(discriminator=discriminator,rna=perturbed_grna,chromosome=chromosome,start=start,end=end,bind_site=bind_site, plot=False)
        heatmap[i,:] = averages
        perturbed_target_means[i] = (abs(np.mean(averages[middle-20:middle+50]) - orig_target_mean)/orig_target_mean)
    
    plt.figure(1)
    x = np.arange(start + 23, end - 23 + 1)
    plt.imshow(heatmap, cmap='inferno', origin='lower', aspect='auto', extent=(min(x), max(x), 0, len(RNA)- perturbation_length + 1))
    plt.colorbar(label='Activity Score')
    plt.xlabel('DNA Position')
    plt.ylabel(f'gRNA Perturbation Index (Length {perturbation_length}, Base {base})')
    plt.yticks(np.arange(0, len(RNA)-perturbation_length + 1))
    plt.grid(axis='y', linestyle='solid', alpha=0.7)
    plt.title('Perturbation Analysis Heatmap')
    plt.show()

    plt.figure(2)
    x = np.arange(0, len(RNA)- perturbation_length + 1)
    plt.bar(x, perturbed_target_means, color='maroon', width=.4)
    plt.xlabel(f'gRNA Perturbation Index (Length {perturbation_length}, Base {base})')
    plt.ylabel('Percent Difference In Activity Score at Target vs Original')
    plt.title('Perturbation Index vs Effect on Target Bind Activity Percent Change')
    plt.show()


def deviation_from_complement_dna(model, dna_sequences, target_sequences):
    deviations = np.zeros_like(target_sequences[0])

    # Ensure X and Y have the same number of sequences
    if len(dna_sequences) != len(target_sequences):
        raise ValueError("The number of DNA sequences and target sequences must be the same")

    for dna_seq, target_seq in zip(dna_sequences, target_sequences):
        # Predict the model output
        prediction = model.predict(np.expand_dims(dna_seq, axis=0))[0]

        # Extract only the DNA part (assuming the first 4 columns represent DNA)
        ohe_dna = dna_seq[:, :4]

        # Calculate the complement (A<->T, C<->G) using vectorized operation
        complement_dna = ohe_dna[:20, [3, 2, 1, 0]]

        # Calculate deviation
        deviation = np.abs(prediction[:20] - complement_dna)
        deviations+= deviation
    
    # Average the deviations
    deviations /= len(dna_sequences)
    debug_print(['shape', deviations.shape, 'deviations:', deviations])

    # Graph deviation per nuceotide per position
    fig, ax = plt.subplots()

    # Plot each line
    ax.plot(deviations[:, 0], label=f'A')
    ax.plot(deviations[:, 1], label=f'C')
    ax.plot(deviations[:, 2], label=f'G')
    ax.plot(deviations[:, 3], label=f'T')
    ax.plot(np.mean(deviations, axis=1), label='Average')

    # Set x-axis and y-axis labels
    ax.set_xlabel('Position')
    ax.set_ylabel('Deviation')

    # Set x-axis ticks at intervals of 5
    ax.set_xticks(np.arange(0, len(deviations), 5))


    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

    return deviations

def generate_examples(model, X, Y, num_examples=10):
    for i in range(num_examples):
        idx = random.randint(0, len(X) - 1)
        x = X[idx]
        y = Y[idx]
        y_pred = model.predict(np.expand_dims(x, axis=0))[0]

        x = ohe_dna_to_sequence(x)
        y = prediction_to_sequence(y)
        y_pred = prediction_to_sequence(y_pred)

        debug_print(['example', i, 'input:', x, 'output:', y, 'prediction:', y_pred])

if __name__ == '__main__':
    os.system('clear')

    graph_roc_curves([
        'metrics/model_metrics.csv'
    ])