import numpy as np
import matplotlib.pyplot as plt

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
        num_seqs=num_seqs)
    
    heatmap = np.zeros((len(rnas), a * 2 - view_length))
    
    for n, (rna, chromosome, start, end) in enumerate(zip(rnas, chromosomes, starts, ends)):
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

 activity_test(
        gan=gan,
        rnas=rnas,
        chromosomes=chromosomes,
        starts=starts,
        ends=ends,
        a=50,
        num_seqs=2)
    
    diffs = []
    for base in ['a', 'g', 'c', 't']:
        diff = perturbation_analysis(
            gan=gan,
            rnas=rnas,
            chromosomes=chromosomes,
            starts=starts,
            ends=ends,
            base=base,
            num_seqs=1,
            a=50
        )[0]
        diffs.append(diff)
    
    x = np.arange(0,4)
    diffs = np.array(diffs).T
    plt.imshow(diffs, cmap='inferno', origin='lower', aspect='auto', extent=(0, 4, 0, 20))
    plt.colorbar(label='Percent Difference')
    plt.xlabel('Replacement Base')
    plt.ylabel(f'gRNA Perturbation Index')
    plt.yticks(np.arange(0, 20))
    plt.xticks(x,['a', 'g', 'c', 't'])
    plt.grid(axis='y', linestyle='solid', alpha=0.7)
    plt.title(f'Percent Difference in Activity at Target for Replacement Bases')
    plt.show()