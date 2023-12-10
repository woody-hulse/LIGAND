from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn import metrics
from scipy import interp
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
import numpy as np

from utils import *
import preprocessing
from tqdm import tqdm


def generate_candidate_grna(gan, rna, chromosome, start, end, a=400, view_length=23, num_seqs=4, plot=True):
    chromosomes = [chromosome for _ in range(num_seqs)]
    starts = [start for _ in range(num_seqs)]
    ends = [end for _ in range(num_seqs)]
    
    if plot:
        fig, axis = plt.subplots(num_seqs, 1, figsize=(8, num_seqs * 2))
        axis[0].set_title('GRNA activity')
        
    X = np.zeros((num_seqs, view_length, 8))
    seq = None
    for n in range(num_seqs):
        seq = preprocessing.fetch_genomic_sequence(chromosome, start, end).lower()
        ohe_seq = np.concatenate([preprocessing.ohe_base(base) for base in seq], axis=0)
        epigenomic_signals = preprocessing.fetch_epigenomic_signals(chromosome, start, end)
        epigenomic_seq = np.concatenate([ohe_seq, epigenomic_signals], axis=1)
        
        X[n] = epigenomic_seq
        
    candidate_grna = gan.generate(X)
    filtered_candidate_grna = []
    candidate_grna_set = set()
    for i in range(candidate_grna.shape[0]):
        grna = preprocessing.str_bases(candidate_grna[i])
        if grna in candidate_grna_set:
            continue
        else:
            filtered_candidate_grna.append(candidate_grna[i])
            candidate_grna_set.add(grna)
    print(candidate_grna_set)
    filtered_candidate_grna = np.array(filtered_candidate_grna)
    
    
    debug_print(['generating candidate grna for', seq, ':'])
    debug_print(['      [correct grna]', preprocessing.str_bases(rna)])
    for grna in candidate_grna:
        debug_print(['     ', preprocessing.str_bases(grna)])
    
    activity_test(gan, filtered_candidate_grna, chromosomes, starts, ends, a, view_length, plot, filtered_candidate_grna.shape[0], True)

def activity_test(gan, rnas, chromosomes, starts, ends, a=400, view_length=23, plot=True, num_seqs=None, test_rna=False):
    if not num_seqs: num_seqs = len(rnas)
    
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    if plot:
        fig, axis = plt.subplots(num_seqs, 1, figsize=(8, num_seqs * 2))
        axis[0].set_title('GRNA activity')
    
    skipped = 0
    skip = []
    
    all_activity_scores = []
    
    X_gen = np.zeros((len(rnas), view_length, 8))
    X = np.zeros((len(rnas), view_length + 2 * a, 8))
    
    for n in range(num_seqs):
        try:
            seq = preprocessing.fetch_genomic_sequence(chromosomes[n], starts[n] - a, ends[n] + a).lower()
            ohe_seq = np.concatenate([preprocessing.ohe_base(base) for base in seq], axis=0)
            epigenomic_signals = preprocessing.fetch_epigenomic_signals(chromosomes[n], starts[n] - a, ends[n] + a)
            epigenomic_seq = np.concatenate([ohe_seq, epigenomic_signals], axis=1)
            if np.isnan(epigenomic_seq).any():
                skip.append(n)
                continue
                
            X[n] = epigenomic_seq
            X_gen[n] = epigenomic_seq[a:a + view_length]
        except Exception as e:
            skip.append(n)
            continue
            
    if test_rna:
        real_Yi = np.array(rnas)
    else:

        pred_Yi = gan.generator(X_gen)
        pred = np.argmax(pred_Yi, axis=2)
        real = np.argmax(rnas, axis=2)
        real_Yi = gan.get_real_Yi(pred_Yi, pred, real)
        
    axis_index = 0
    for n, (rna, chromosome, start, end) in enumerate(zip(rnas, chromosomes, starts, ends)):
        if n >= num_seqs + skipped: continue
        if n in skip:
            skipped += 1
            continue
        preprocessing.debug_print(['running activity test on', preprocessing.str_bases(rna)])
        bind_site = start
        start -= a
        end += a
        
        activity_scores = []
        step = 1
        for i in range(0, end - start - view_length, step):
            if gan.discriminator.name == 'conv_discriminator' or gan.discriminator.name == 'critic_transformer_1':
                activity_score = gan.discriminator([
                    np.expand_dims(X[n][i:i+view_length], axis=0), 
                    np.expand_dims(real_Yi[n], axis=0)
                ])
                for _ in range(step):
                    activity_scores.append(activity_score[0][0].numpy())
            else:
                pass # finish
        activity_scores = np.array(activity_scores)
        moving_averages = activity_scores[23:]# moving_average(activity_scores, 100)
        all_activity_scores.append(activity_scores)
            
        if plot:
            x = np.arange(start + view_length, end - view_length + 4)[:len(moving_averages)]
            axis[axis_index].plot(x, moving_averages, label=preprocessing.str_bases(rna))
            axis[axis_index].axvline(x=bind_site, color='orange', linestyle='dotted', label='bind site')
            axis[axis_index].legend()
        
        axis_index += 1
    
    if plot:
        plt.xlabel('genomic position')
        plt.ylabel('predicted activity')
        plt.show()
    
    return all_activity_scores

def perturbation_analysis(gan, rnas, chromosomes, starts, ends, base, a=400, view_length=23, num_seqs=None):
    if not num_seqs: num_seqs = len(rnas)
    
    skipped = 0
    skip = []
    
    X_gen = np.zeros((len(rnas), view_length, 8))
    X = np.zeros((len(rnas), view_length + 2 * a, 8))
    for n in range(num_seqs):
        try:
            seq = preprocessing.fetch_genomic_sequence(chromosomes[n], starts[n] - a, ends[n] + a).lower()
            ohe_seq = np.concatenate([preprocessing.ohe_base(base) for base in seq], axis=0)
            epigenomic_signals = preprocessing.fetch_epigenomic_signals(chromosomes[n], starts[n] - a, ends[n] + a)
            epigenomic_seq = np.concatenate([ohe_seq, epigenomic_signals], axis=1)
            if np.isnan(epigenomic_seq).any():
                skip.append(n)
                continue
            
            X[n] = epigenomic_seq
            X_gen[n] = epigenomic_seq[a:a + view_length]
        except:
            skip.append(n)
            continue

    pred_Yi = gan.generator(X_gen)
    pred = np.argmax(pred_Yi, axis=2)
    real = np.argmax(rnas, axis=2)
    real_Yi = gan.get_real_Yi(pred_Yi, pred, real)
    axis_index = 0
    
    original = activity_test(
        gan=gan,
        rnas=rnas,
        chromosomes=chromosomes,
        starts=starts,
        ends=ends,
        a=a,
        view_length=view_length,
        plot=False,
        num_seqs=num_seqs)
    
    heatmap = np.zeros((len(rnas), a * 2 - view_length))
    
    for n, (rna, chromosome, start, end) in enumerate(zip(rnas, chromosomes, starts, ends)):
        if n >= num_seqs + skipped: continue
        if n in skip:
            skipped += 1
            continue
        
        start -= a
        end += a
        
        for i in range(len(rna)):
            perturbed_grna = real_Yi[n]
            base_index = np.argmax(preprocessing.ohe_base(base))
            perturbed_index = np.argmax(perturbed_grna[i])
            perturbed_grna[base_index], perturbed_grna[perturbed_index] = perturbed_grna[perturbed_index], perturbed_grna[base_index]
            
            perturbed_activity_scores = []
            for j in range(end - start - view_length):
                if gan.discriminator.name == 'conv_discriminator':
                    activity_score = gan.discriminator([
                        np.expand_dims(X[n][j:j+view_length], axis=0), 
                        np.expand_dims(perturbed_grna, axis=0)
                    ])
                    perturbed_activity_scores.append(activity_score.numpy()[0][0])
            heatmap[i, :] = np.array(perturbed_activity_scores)[view_length - 1:]
        
        axis_index += 1
    
        x = np.arange(start + 23, end - 23 + 1)
        plt.imshow(heatmap, cmap='inferno', origin='lower', aspect='auto', extent=(min(x), max(x), 0, len(rna)))
        plt.colorbar(label='Activity Score')
        plt.xlabel('DNA Position')
        plt.ylabel(f'gRNA Perturbation Index (Length {1}, Base {base})')
        plt.yticks(np.arange(0, len(rna)))
        plt.grid(axis='y', linestyle='solid', alpha=0.7)
        plt.title('Perturbation Analysis Heatmap')
        plt.show()
        
    

'''
def perturbation_analysis(gan, chromosome, start, end, bind_site, rna, perturbation_length, base='N'):
    RNA = rna
    heatmap = np.zeros((len(RNA) - perturbation_length + 1, 777))
    _,original = activity_test(gan=gan,rna=RNA,chromosome=chromosome,start=start,end=end,bind_site=bind_site, plot=False)
    middle = len(original)//2
    orig_target_mean = np.mean(original[middle-20:middle+50])
    perturbed_target_means = np.zeros(len(RNA) - perturbation_length + 1)
    
    for i in range(0, len(RNA)-perturbation_length + 1):
        perturbed_grna = list(RNA)
        perturbed_grna[i:i + perturbation_length] = base * perturbation_length
        perturbed_grna = ''.join(perturbed_grna)
        _, averages = activity_test(gan=gan,rna=perturbed_grna,chromosome=chromosome,start=start,end=end,bind_site=bind_site, plot=False)
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
'''

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
        # file is of form 'models/mlp_gan/roc.csv'
        model_name = file.split('/')[1]
        plt.plot(df['fpr'], df['tpr'], label=f'{model_name} (AUC = {df["auc"][0]:.4f})', lw=2, alpha=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


def deviation_from_complement_dna(model, dna_sequences, target_sequences):
    deviations = np.zeros_like(target_sequences[0])

    # Ensure X and Y have the same number of sequences
    if len(dna_sequences) != len(target_sequences):
        raise ValueError("The number of DNA sequences and target sequences must be the same")

    # Add tqdm progress bar
    for dna_seq, target_seq in tqdm(zip(dna_sequences, target_sequences), total=len(dna_sequences), desc='Calculating deviations'):
        # Predict the model output
        prediction = model.predict(np.expand_dims(dna_seq, axis=0))[0]

        # Extract only the DNA part (assuming the first 4 columns represent DNA)
        ohe_dna = dna_seq[:, :4]

        # Calculate the complement (A<->T, C<->G) using vectorized operation
        complement_dna = ohe_dna[:20, [3, 2, 1, 0]]

        # Calculate deviation
        deviation = np.abs(prediction[:20] - complement_dna)
        deviations += deviation
    
    # Average the deviations
    deviations /= len(dna_sequences)

    # Concat the average deviation to the end of the matrix
    deviations = np.concatenate([deviations, np.mean(deviations, axis=1, keepdims=True)], axis=1)
    debug_print(['shape', deviations.shape, 'deviations:', deviations])


    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))  # You can adjust the figure size as needed

    # Create a heatmap
    sns.heatmap(deviations, annot=True, ax=ax, cmap='viridis')

    # Set x-axis and y-axis labels
    ax.set_xlabel('Nucleotides')
    ax.set_ylabel('Position')

    # Optionally, set the x-axis ticks to represent nucleotides
    # ax.set_xticks(range(5))  # Set 5 tick locations
    ax.set_xticklabels(['A', 'C', 'G', 'T', 'Average'])

    # Optionally, adjust the y-axis ticks to show positions
    ax.set_yticklabels(np.arange(1, len(deviations) + 1))

    # Show the plot
    plt.show()

    return deviations

if __name__ == '__main__':
    os.system('clear')

    graph_roc_curves([
        'models/mlp_gan/roc.csv',
        'models/conv_gan/roc.csv',
        'models/trans_conv_gan/roc.csv',
        'models/trans_gan/roc.csv',
        'models/trans_conv_gan2/roc.csv',
    ])
    

    