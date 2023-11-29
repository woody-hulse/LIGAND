import csv
import os
from tqdm import tqdm
import datetime

import numpy as np
import pandas as pd

from Bio import SeqIO
import pyBigWig

H3K4ME3_PATH = "data/h3k4me.bigWig" # https://drive.google.com/drive/u/0/folders/1nycERZkXh5Qiyy8HQE3Hk0pOez_UFECw
RRBS_PATH = "data/methyl.bigBed"
DNASE_PATH = "data/dnase.bigWig"
CTCF_PATH = "data/ctcf.bigWig"

REFERENCE_GENOME_PATH = 'data/GCF_000001405.26_GRCh38_genomic.fna' # https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.26/
GRNA_PATH = 'data/hg_guide_info.csv' # https://www.ncbi.nlm.nih.gov/genome/guide/human/

GENOME_SEQUENCES = {}
def debug_print(statements=[], end='\n'):
    ct = datetime.datetime.now()
    print('[', str(ct)[:19], '] ', sep='', end='')
    for statement in statements:
        print(statement, end=' ')
    print(end=end)


def read_genome(path=REFERENCE_GENOME_PATH):
    debug_print(['loading genomic data from', path])
    for record in SeqIO.parse(path, 'fasta'):
        GENOME_SEQUENCES[record.id.split('.')[0]] = record.seq


def fetch_genomic_sequence(chromosome, start, end, a=0):
    chromosome_id = f'NC_0000{chromosome}'
    genome_sequence = GENOME_SEQUENCES.get(chromosome_id)
    if genome_sequence:
        return str(genome_sequence[start - a:end + a + 1]) # start - 1
    else:
        raise ValueError(f'chromosome {chromosome_id} not found in the genome file.')
    

def fetch_epigenomic_signals(chromosome, start, end, a=0):
    signals = np.zeros((end - start + 1 + 2 * a, 4))
    
    chromosome = 'chr' + str(int(chromosome))
    
    h3k4me_file = pyBigWig.open(H3K4ME3_PATH)
    rrbs_file = pyBigWig.open(RRBS_PATH)
    dnase_file = pyBigWig.open(DNASE_PATH)
    ctcf_file = pyBigWig.open(CTCF_PATH)
    
    # print(chromosome, start - a, end  + a + 1)
    
    def set_signal(index, entries):
        if not entries:
            return
        for e in entries:
            read_start = e[0]
            read_end = e[1]
            string_vals = e[2].split('\t')
            val = 0 
            if string_vals[0].isnumeric():
                if float(string_vals[0]) > 0: val = 1
                else: val = 0 
                # val = float(string_vals[0]) / 1000
            else:
                if float(string_vals[1]) > 0: val = 1
                else: val = 0 
                # val = float(string_vals[1]) / 1000
            signals[read_start-start:read_end-start, index] = val
    
    h3k4me_vals = np.array(h3k4me_file.values(chromosome, start - a, end + a + 1))
    signals[:, 2] = h3k4me_vals if not h3k4me_vals.any() == None else signals[:, 2]
    set_signal(1, rrbs_file.entries(chromosome, start - a, end + a + 1))
    dnase_vals = np.array(dnase_file.values(chromosome, start - a, end + a + 1))
    signals[:, 2] = dnase_vals if not dnase_vals.any() == None else signals[:, 2]
    ctcf_vals = np.array(ctcf_file.values(chromosome, start - a, end + a + 1))
    signals[:, 2] = ctcf_vals if not ctcf_vals.any() == None else signals[:, 2]
    
    h3k4me_file.close()
    rrbs_file.close()
    dnase_file.close()
    ctcf_file.close()
    
    return signals


def filter_bases_lists(bases1, bases2):
    debug_print(['filtering base sequences'])
    filter_bases1 = []
    filter_bases2 = []
    for base1, base2 in zip(bases1, bases2):
        set1 = set(list(base1.lower()))
        set2 = set(list(base2.lower()))
        if 'n' not in set1 and 'n' not in set2:
            filter_bases1.append(base1)
            filter_bases2.append(base2)

    return filter_bases1, filter_bases2


def ohe_base(base):
    ohe = np.zeros((1, 4))
    if base == 'a': ohe[0, 0] = 1
    if base == 'g': ohe[0, 1] = 1
    if base == 'c': ohe[0, 2] = 1
    if base == 't': ohe[0, 3] = 1
    return ohe


def ohe_bases(bases_lists):
    debug_print(['one-hot encoding bases'])
    ohe = np.zeros((len(bases_lists), len(bases_lists[0]), 4))
    for i, bases_list in enumerate(bases_lists):
        for j, base in enumerate(bases_list):
            if j >= len(bases_lists[0]): continue
            ohe[i, j] = ohe_base(base)
    return ohe


def tokenize_bases(bases_lists):
    debug_print(['tokenizing encoding bases'])
    tokenized = np.zeros((len(bases_lists), len(bases_lists[0])))
    for i, bases_list in enumerate(bases_lists):
        for j, base in enumerate(bases_list):
            if j >= len(bases_lists[0]): continue
            if base == 'a': tokenized[i, j] = 0
            if base == 'g': tokenized[i, j] = 1
            if base == 'c': tokenized[i, j] = 2
            if base == 't': tokenized[i, j] = 3
    return tokenized


def batch_data(data, batch_size):
    debug_print(['batching data'])
    output_length = data.shape[0] // batch_size
    batched_data = np.zeros((output_length, batch_size, data.shape[1], data.shape[2]))
    for index in range(output_length * batch_size):
        i = index // batch_size
        j = index % batch_size
        batched_data[i][j] = data[index]

    return batched_data


def train_val_test_split(data, train=0.7, val=0.2, test=0.1):
    length = len(data)
    return data[:int(train*length)], \
           data[int(train*length):int((train + val)*length)], \
           data[int((train + val)*length):int((train + val + test)*length)]


def load_data(seqs_path='data/seqs.npy', grna_path='data/grna.npy'):
    debug_print(['loading preprocessed data'])
    seqs, grna = np.load(seqs_path), np.load(grna_path)
    debug_print([
        'data shape:', 
        '\n                  DNA:  ', seqs.shape, 
        '\n                  gRNA: ', grna.shape])
    return seqs, grna


def get_train_test(df, length=1e4):
    read_genome()

    grna_list = []
    seqs_list = []
    exclude = set(['0X', '0Y', '0M'])
    bases = set(['a', 'c', 'g', 't'])
    debug_print(['locating corresponding genome sequences'])
    for index, row in tqdm(df.iterrows()):
        if len(grna_list) >= length: break
        chromosome = row[3][3:].zfill(2)
        if chromosome in exclude: continue
        if row[0][0] == 'N': continue
        start, end = row[4], row[5]
        try:
            seq = fetch_genomic_sequence(chromosome, start, end).lower()
            rna = row[0].lower()
            if set(list(seq)).union(bases) == set(list(seq)) and \
               set(list(rna)).union(bases) == set(list(rna)):
                   
                ohe_seq = np.concatenate([ohe_base(base) for base in seq], axis=0)
                epigenomic_signals = fetch_epigenomic_signals(chromosome, start, end)
                if not epigenomic_signals.shape == (23, 4): continue
                epigenomic_seq = np.concatenate([ohe_seq, epigenomic_signals], axis=1)
                if np.isnan(epigenomic_seq).any(): continue
                
                ohe_rna = np.concatenate([ohe_base(base) for base in rna], axis=0)
                seqs_list.append(epigenomic_seq)
                grna_list.append(ohe_rna)
        except:
            # print(f'chromosome {chromosome} not found in the genome file.')
            pass
    seqs = np.array(seqs_list)
    grna = np.array(grna_list)
    # grna = ohe_bases(grna_list)
    # seqs = ohe_bases(seqs_list)
    debug_print([
        'data shape:', 
        '\n                  DNA:  ', seqs.shape, 
        '\n                  gRNA: ', grna.shape])

    return seqs, grna


def get_discriminator_train_test(seqs, grna):
    debug_print(['generating synthetic data'])
    length = seqs.shape[0]

    synthetic_seqs = seqs[np.random.permutation(length)]
    synthetic_grna = grna[np.random.permutation(length)]

    seqs = np.concatenate([seqs, seqs])
    grna = np.concatenate([grna, synthetic_grna])

    indices = np.random.permutation(length * 2)
    X = [seqs[indices], grna[indices]]
    Y = np.array([1 for _ in range(length)] + [0 for _ in range(length)])
    Y = Y[indices]
    
    return X, Y


def extract_data(path=GRNA_PATH):
    debug_print(['loading gRNA data from', path])
    df = pd.read_csv(path)
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    return shuffled_df