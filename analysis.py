from sklearn.metrics import roc_curve, auc
from scipy import interp
import os
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
import numpy as np
import tensorflow as tf

from utils import *
import preprocessing
from tqdm import tqdm


def generate_candidate_grna(gan, rna, chromosome, start, end, a=400, view_length=23, num_seqs=4, plot=True):
    debug_print(['generating candidate grna for', chromosome, start, ':', end])
    
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

def activity_test(gan, rnas, chromosomes, starts, ends, a=400, view_length=23, plot=True, num_seqs=None, test_rna=False, return_sep=False, experimental_efficacies=None):
    if not num_seqs: num_seqs = len(rnas)
    
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    if plot:
        fig, axis = plt.subplots(num_seqs, 1, figsize=(8, num_seqs * 2))
        axis[0].set_title('GRNA activity')
    
    skipped = 0
    skip = []
    
    all_activity_scores = np.zeros(2 * a - 1)
    sep_activity_scores = []
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
        all_activity_scores += np.array(activity_scores)
        sep_activity_scores.append(activity_scores)
        if plot:
            x = np.arange(start + view_length, end - view_length + 4)[:len(moving_averages)]
            axis[axis_index].plot(x, moving_averages, label=preprocessing.str_bases(rna))
            if experimental_efficacies is None:
                axis[axis_index].axvline(x=bind_site, color='orange', linestyle='dotted', label='bind site')
            else:
                axis[axis_index].scatter(x=bind_site, y=experimental_efficacies[n], color='red', label='experimental efficacy')
            axis[axis_index].legend()
        
        axis_index += 1
    
    if plot:
        plt.xlabel('genomic position')
        plt.ylabel('predicted activity')
        plt.show()
    
    if return_sep:
        return sep_activity_scores
    return all_activity_scores / axis_index

def candidate_grna_range(gan, chromosome, start, end, a=400, view_length=23, num_seqs=5, plot=True):
    
    length = 2 * a + end - start - view_length
    
    debug_print(['generating best grna candidates for chromosome', chromosome, start, ':', end])
    
    seq = preprocessing.fetch_genomic_sequence(chromosome, start - a, end + a).lower()
    ohe_seq = np.concatenate([preprocessing.ohe_base(base) for base in seq], axis=0)
    epigenomic_signals = preprocessing.fetch_epigenomic_signals(chromosome, start - a, end + a)
    epigenomic_seq = np.concatenate([ohe_seq, epigenomic_signals], axis=1)
    X = epigenomic_seq
    
    best_seqs = [(0, np.inf, '', np.zeros(length)) for _ in range(num_seqs)]
    for i in tqdm(range(length)):
        X_gen = X[i:i + view_length]
        Y_gen = gan.generate(np.expand_dims(X_gen, axis=0))
        activity_scores = np.zeros(length)
        target_activity = np.zeros(length)
        target_activity[i] = 1
        
        for j in range(length):
            X_disc = X[j:j + view_length]
            activity_score = gan.discriminator([
                np.expand_dims(X_disc, axis=0),
                Y_gen
            ])
            activity_scores[j] = activity_score
        
        loss = tf.keras.losses.categorical_crossentropy(target_activity, activity_scores)
        
        Y_gen_ohe = tf.one_hot(tf.math.argmax(Y_gen, axis=2), 4, axis=2)[0]
        tup = (i, loss, preprocessing.str_bases(Y_gen_ohe), activity_scores)
        
        for k in range(num_seqs):
            if best_seqs[k][1] > tup[1]:
                best_seqs[k], tup = tup, best_seqs[k]
    
    
    if plot:
        fig, axis = plt.subplots(num_seqs, 1, figsize=(8, num_seqs * 2))
        axis[0].set_title('Best gRNA found for ' + chromosome + ' ' + str(start) + ':' + str(end))
        for i in range(num_seqs):
            x = np.arange(length)
            axis[i].plot(x, best_seqs[i][3], label=best_seqs[i][2])
            axis[i].legend()
        plt.xlabel('genomic position')
        plt.ylabel('predicted activity')
        plt.show()
    
    return best_seqs

def perturbation_analysis(gan, rnas, chromosomes, starts, ends, base, a=400, view_length=23, num_seqs=None, plot=True):
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
    cumulative_percent_diff = []

    original = activity_test(
        gan=gan,
        rnas=rnas,
        chromosomes=chromosomes,
        starts=starts,
        ends=ends,
        a=a,
        view_length=view_length,
        plot=False,
        num_seqs=num_seqs,
        return_sep=True
        )
    
    heatmap = np.zeros((len(rnas), a * 2 - view_length))
    
    for n, (rna, chromosome, start, end) in enumerate(zip(rnas, chromosomes, starts, ends)):
        debug_print(['running perturbation analysis on', preprocessing.str_bases(rna)])
        if n >= num_seqs + skipped: continue
        if n in skip:
            skipped += 1
            continue
        
        start -= a
        end += a
        percent_diff = []
        heatmap = np.zeros((len(rna), a * 2 - view_length))
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
            heatmap[i] = np.array(perturbed_activity_scores)[view_length - 1:]
            percent_diff.append(((heatmap[i][a-view_length+1] - original[n][a])/original[n][a])*100)
        cumulative_percent_diff.append(percent_diff)
        axis_index += 1    

        if plot:

            plt.figure(figsize=(20, 7))
            plt.subplot(1, 2, 1)
            x = np.arange(start + 23, end - 23 + 1)
            plt.imshow(heatmap, cmap='inferno', origin='lower', aspect='auto', extent=(min(x), max(x), 0, len(rna)))
            plt.colorbar(label='Activity Score')
            plt.xlabel('DNA Position')
            plt.ylabel(f'gRNA Perturbation Index')
            plt.yticks(np.arange(0, len(rna)))
            plt.grid(axis='y', linestyle='solid', alpha=0.7)
            str_rna = preprocessing.str_bases(rna)
            plt.title(f'Heatmap')

            plt.subplot(1, 2, 2)
            x = np.arange(0, len(rna))
            plt.bar(x, percent_diff, color='maroon', width=.4)
            plt.axhline(0, color='black', linewidth=1, linestyle='solid')
            plt.xlabel(f'gRNA Perturbation Index')
            plt.ylabel('Percent Difference In Activity Score at Target vs Original')
            plt.title(f'Percent Difference')
            plt.xticks(x) 

            plt.suptitle(f'Perturbation Analysis of {str_rna} with base {base} and length {1}')
            plt.show()
    
    return cumulative_percent_diff

def perturbation_map(gan, rnas, chromosomes, starts, ends, view_length=23, num_seqs=4):
    X = np.zeros((len(rnas), view_length, 8))
    
    skipped = 0
    skip = []
    
    for n in range(num_seqs):
        try:
            seq = preprocessing.fetch_genomic_sequence(chromosomes[n], starts[n], ends[n]).lower()
            ohe_seq = np.concatenate([preprocessing.ohe_base(base) for base in seq], axis=0)
            epigenomic_signals = preprocessing.fetch_epigenomic_signals(chromosomes[n], starts[n], ends[n])
            epigenomic_seq = np.concatenate([ohe_seq, epigenomic_signals], axis=1)
            if np.isnan(epigenomic_seq).any():
                skip.append(n)
                continue
                
            X[n] = epigenomic_seq
        except Exception as e:
            skip.append(n)
            continue
            
    pred_Yi = gan.generator(X)
    pred = np.argmax(pred_Yi, axis=2)
    real = np.argmax(rnas, axis=2)
    real_Yi = gan.get_real_Yi(pred_Yi, pred, real)
    
    for n, (rna, chromosome, start, end) in enumerate(zip(rnas, chromosomes, starts, ends)):
        if n >= num_seqs + skipped: continue
        if n in skip:
            skipped += 1
            continue
        
        heatmap = np.zeros((len(rna), 4))
        
        base_activity_score = gan.discriminator([
                        np.expand_dims(X[n], axis=0), 
                        np.expand_dims(real_Yi[n], axis=0)
                    ])
        
        Yn = real_Yi[n]
        for i in range(len(rna)):
            argmax = np.argmax(Yn[i])
            for j in range(4):
                if j == argmax:
                    heatmap[i, j] = 0
                    continue
                else:
                    perturbed_Yn = copy.deepcopy(Yn)
                    perturbed_Yn[i, j], perturbed_Yn[i, argmax] = perturbed_Yn[i, argmax], perturbed_Yn[i, j]
                    perturbed_activity_score = gan.discriminator([
                        np.expand_dims(X[n], axis=0), 
                        np.expand_dims(perturbed_Yn, axis=0)
                    ])
                    heatmap[i, j] = perturbed_activity_score - base_activity_score
        
        # inverse heatmap axis
        heatmap = np.flip(heatmap, axis=1)

        plt.figure(1)
        plt.imshow(heatmap, cmap='inferno', origin='lower', aspect='auto', extent=(0, 20, 0, 4))
        plt.colorbar(label='Activity Score')
        plt.xlabel('Replacement Base')
        plt.ylabel(f'gRNA Perturbation Index')
        plt.yticks(np.arange(0, 20))
        plt.xticks([0, 1, 2, 3],['a', 'g', 'c', 't'])
        plt.grid(axis='y', linestyle='solid', alpha=0.7)
        plt.title('Perturbation Analysis Heatmap')
        plt.show()

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

def graph_metrics(files):
    sns.set_theme(style="darkgrid")

    # Gen loss
    for file in files:
        df = pd.read_csv(file)
        # file is of form 'models/mlp_gan/roc.csv'
        model_name = file.split('/')[1]
        plt.plot(df['gen_losses'], label=f'{model_name}', lw=2, alpha=1)
    plt.xlabel('Epoch')
    plt.ylabel('Generator Loss')
    plt.title('Generator Loss')
    plt.legend(loc="lower right")
    plt.show()

    # Disc loss
    for file in files:
        df = pd.read_csv(file)
        # file is of form 'models/mlp_gan/roc.csv'
        model_name = file.split('/')[1]
        plt.plot(df['disc_losses'], label=f'{model_name}', lw=2, alpha=1)
    plt.xlabel('Epoch')
    plt.ylabel('Discriminator Loss')
    plt.title('Discriminator Loss')
    plt.legend(loc="lower right")
    plt.show()

    # Total loss
    for file in files:
        df = pd.read_csv(file)
        # file is of form 'models/mlp_gan/roc.csv'
        model_name = file.split('/')[1]
        plt.plot(df['gen_losses'] + df['disc_losses'], label=f'{model_name}', lw=2, alpha=1)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss')
    plt.legend(loc="lower right")
    plt.show()

    # Gen real loss
    for file in files:
        df = pd.read_csv(file)
        # file is of form 'models/mlp_gan/roc.csv'
        model_name = file.split('/')[1]
        plt.plot(df['gen_real_losses'], label=f'{model_name}', lw=2, alpha=1)
    plt.xlabel('Epoch')
    plt.ylabel('Generator Real Loss')
    plt.title('Generator Real Loss')
    plt.legend(loc="lower right")
    plt.show()

    # Gen accuracy
    for file in files:
        df = pd.read_csv(file)
        # file is of form 'models/mlp_gan/roc.csv'
        model_name = file.split('/')[1]
        plt.plot(df['gen_accuracies'], label=f'{model_name}', lw=2, alpha=1)
    plt.xlabel('Epoch')
    plt.ylabel('Generator Accuracy')
    plt.title('Generator Accuracy')
    plt.legend(loc="lower right")
    plt.show()

    # Disc accuracy
    for file in files:
        df = pd.read_csv(file)
        # file is of form 'models/mlp_gan/roc.csv'
        model_name = file.split('/')[1]
        plt.plot(df['disc_accuracies'], label=f'{model_name}', lw=2, alpha=1)
    plt.xlabel('Epoch')
    plt.ylabel('Discriminator Accuracy')
    plt.title('Discriminator Accuracy')
    plt.legend(loc="lower right")
    plt.show()

def deviation_from_complement_dna(model, dna_sequences):
    deviations = np.zeros((len(dna_sequences), 20, 4))

    # Ensure X and Y have the same number of sequences
    for dna_seq, _ in tqdm(dna_sequences, total=len(dna_sequences), desc='Calculating deviations'):
        prediction = model.predict(np.expand_dims(dna_seq, axis=0))[0]

        ohe_dna = dna_seq[:, :4]

        # Calculate the complement (A<->T, C<->G) using vectorized operation
        complement_dna = ohe_dna[:20, [3, 2, 1, 0]]

        deviation = (prediction - complement_dna)
        deviations += deviation

    # Normalize
    deviations /= len(dna_sequences)

    # Concatenate the average deviations to the end of the matrix
    average_per_position = np.mean(deviations, axis=1, keepdims=True)
    graph_deviations = np.concatenate([deviations, average_per_position], axis=1)

    average_per_nucleotide = np.mean(graph_deviations, axis=0, keepdims=True)
    graph_deviations = np.concatenate([graph_deviations, average_per_nucleotide], axis=0)
    debug_print(['shape', graph_deviations.shape])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))  # You can adjust the figure size as needed

    # Create a heatmap
    sns.heatmap(graph_deviations, annot=True, ax=ax, cmap='viridis')

    # Set x-axis and y-axis labels
    ax.set_xlabel('Nucleotides')
    ax.set_ylabel('Position')

    ax.set_xticklabels(['A', 'C', 'G', 'T', 'Average'])
    position_labels = list(np.arange(1, len(deviations) + 1)) + ['Avg per Nucleotide']
    ax.set_yticklabels(position_labels)
    # Show the plot
    plt.show()

    return deviations

def deviation_from_complement_dna_single_strand(model, dna_seq):

    prediction = model.predict(np.expand_dims(dna_seq, axis=0))[0]

    ohe_dna = dna_seq[:, :4]

    # Calculate the complement (A<->T, C<->G) using vectorized operation
    complement_dna = ohe_dna[:20, [3, 2, 1, 0]]

    deviation = (prediction - complement_dna)
    average_per_nucleotide = np.mean(deviation, axis=0, keepdims=True)
    graph_deviations = np.concatenate([deviation, average_per_nucleotide], axis=0)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))  # You can adjust the figure size as needed

    # Create a heatmap
    sns.heatmap(graph_deviations, annot=True, ax=ax, cmap='viridis')

    # Set x-axis and y-axis labels
    ax.set_xlabel('Nucleotides')
    ax.set_ylabel('Position')

    ax.set_xticklabels(['A', 'C', 'G', 'T'])
    position_labels = list(np.arange(1, len(deviation) + 1)) + ['Avg']
    ax.set_yticklabels(position_labels)
    # Show the plot
    plt.show()

    return deviation

def complement_activity_test(gan, chromosome, start, end, a=400, view_length=23, plot=True, num_seqs=None, test_rna=False):    
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    
    seq = preprocessing.fetch_genomic_sequence(chromosome, start - a, end + a).lower()
    ohe_seq = np.concatenate([preprocessing.ohe_base(base) for base in seq], axis=0)
    compl_seq = ohe_seq[:, [3, 2, 1, 0]]
    epigenomic_signals = preprocessing.fetch_epigenomic_signals(chromosome, start - a, end + a)
    epigenomic_seq = np.concatenate([ohe_seq, epigenomic_signals], axis=1)
                
    start -= a
    end += a
    
    generator_scores = []
    complement_scores = []
    step = 1
    for i in range(0, end - start - view_length, step):
        epi_seq = epigenomic_seq[i:i+view_length]
        if gan.discriminator.name == 'conv_discriminator' or gan.discriminator.name == 'critic_transformer_1':
            generator_score = gan.discriminator([
                np.expand_dims(epi_seq, axis=0), 
                gan.generator(np.expand_dims(epi_seq, axis=0))
            ])
            
            complement_score = gan.discriminator([
                np.expand_dims(epi_seq, axis=0), 
                np.expand_dims(compl_seq[i:i+view_length - 3], axis=0)
            ])
            for _ in range(step):
                generator_scores.append(generator_score[0][0].numpy())
                complement_scores.append(complement_score[0][0].numpy())
        else:
            pass # finish

    generator_scores = np.array(generator_scores)
    complement_scores = np.array(complement_scores)
    gen_moving_averages = generator_scores[23:]# moving_average(generator_scores, 100)
    compl_moving_averages = complement_scores[23:]# moving_average(complement_scores, 100)

    x = np.arange(start + view_length, end - view_length + 4)[:len(gen_moving_averages)]
    plt.plot(x, gen_moving_averages, label='generated')
    plt.plot(x, compl_moving_averages, label='complement')
    plt.legend()    
   
    plt.xlabel('genomic position')
    plt.ylabel('predicted activity')
    plt.show()
    
    return (gen_moving_averages, compl_moving_averages)

def validate_against_efficacies(gan, plot = True):
    preprocessing.populate_efficacy_map()
    efficacies = []
    for ((sgRNA, chromosome, start, end), efficacy) in preprocessing.EFFICACY_MAP.items():
        seq = preprocessing.fetch_genomic_sequence(chromosome, start, end)
        try:
            seq = np.concatenate([ohe_base(base) for base in seq], axis=0)
        except:
            # debug_print(['skipping', chromosome, start, end])
            continue
        epi = preprocessing.fetch_epigenomic_signals(chromosome, start, end)
        seq = np.concatenate([seq, epi], axis=1)
        seq = np.expand_dims(seq, axis=0)
        grna = np.concatenate([ohe_base(base) for base in sgRNA], axis=0)
        grna = np.expand_dims(grna, axis=0)
        X = [seq, grna]
        
        predicted = gan.discriminator(X)[0][0].numpy()
        efficacies.append((chromosome, start, efficacy, predicted))

    if plot:
        debug_print(['plotting efficacy differences for n seqs:', len(efficacies)])
        df = pd.DataFrame(efficacies, columns=['Chromosome', 'Start', 'TrueEfficacy', 'PredictedEfficacy'])

        # # Get unique chromosomes
        # chromosomes = df['Chromosome'].unique()

        # # Plot data for each chromosome
        # for i, chrom in enumerate(chromosomes):
        #     chrom_df = df[df['Chromosome'] == chrom]

        #     fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        #     axes.set_title(f'Chromosome {chrom}')
        #     axes.set_xlabel('Position')
        #     axes.set_ylabel('Efficacy')
        #     axes.plot(chrom_df['Start'], chrom_df['TrueEfficacy'], label='True Efficacy')
        #     axes.plot(chrom_df['Start'], chrom_df['PredictedEfficacy'], label='Predicted Efficacy')
        #     axes.legend()
        #     plt.show()

        # Calculate the difference in efficacy
        df['EfficacyDifference'] = df['PredictedEfficacy'] - df['TrueEfficacy']

        # Create a histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df['EfficacyDifference'], bins=20, color='skyblue', edgecolor='black')

        # Adding labels and title
        plt.xlabel('Difference in Efficacy (Predicted - True)')
        plt.ylabel('Number of Occurrences')
        plt.title('Distribution of Efficacy Differences')

        # Show the plot
        plt.show()

    return efficacies

def validation_activity_map(gan, num_seqs=4):
    preprocessing.populate_efficacy_map()

    rnas = []
    chromosomes = []
    starts = []
    ends = []
    efficacies = []
    for ((sgRNA, chromosome, start, end), efficacy) in preprocessing.EFFICACY_MAP.items():
        if len(rnas) >= num_seqs:
            break
        rnas.append(ohe_bases(sgRNA)[:-3])
        chromosomes.append(chromosome)
        starts.append(start)
        ends.append(end)
        efficacies.append(efficacy)
    
    rnas = np.array(rnas).reshape((len(rnas), 20, 4))

    print(rnas.shape)

    activity_test(gan, rnas, chromosomes, starts, ends, experimental_efficacies=efficacies, num_seqs=num_seqs)


if __name__ == '__main__':
    os.system('clear')

    graph_roc_curves([
        'models/mlp_gan/roc.csv',
        'models/conv_gan/roc.csv',
        'models/trans_gan/roc.csv',
        'models/trans_conv_gan2/roc.csv',
        'models/trans_gan4/roc.csv'
    ])

    graph_metrics([
       'models/mlp_gan/metrics.csv',
        'models/conv_gan/metrics.csv',
        'models/trans_gan/metrics.csv',
        'models/trans_conv_gan2/metrics.csv',
        'models/trans_gan4/metrics.csv'
    ])

    

    