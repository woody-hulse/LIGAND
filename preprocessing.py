import csv
import os
from tqdm import tqdm
import datetime

import numpy as np
import pandas as pd

from Bio import SeqIO

GENOME_SEQUENCES_PATH = 'GRCh37_latest_genomic.fna'     # https://www.ncbi.nlm.nih.gov/genome/guide/human/
H3K4ME3_PATH = ""
RRBS_PATH = ""
DNASE_PATH = ""
CTCF_PATH = ""
DATA_PATH = 'hg_guide_info.csv'

GENOME_SEQUENCES = {}
def debug_print(statements=[], end='\n'):
    ct = datetime.datetime.now()
    print('[', str(ct)[:19], '] ', sep='', end='')
    for statement in statements:
        print(statement, end=' ')
    print(end=end)


def read_genome(df, path='GRCh37_latest_genomic.fna'):
    tm = SeqIO.parse(path, 'fasta')
    debug_print(['dna sequence imported from', path])


def fetch_genome_sequence(chromosome, start, end, a=0):
    chromosome_id = f'NC_0000{chromosome}'
    genome_sequence = GENOME_SEQUENCES.get(chromosome_id)
    if genome_sequence:
        return str(genome_sequence[start - a:end + a + 1]) # start - 1
    else:
        raise ValueError(f'chromosome {chromosome_id} not found in the genome file.')


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


def ohe_bases(bases_lists):
    debug_print(['one-hot encoding bases'])
    ohe = np.zeros((len(bases_lists), len(bases_lists[0]), 4))
    for i, bases_list in enumerate(bases_lists):
        for j, base in enumerate(bases_list):
            if j >= len(bases_lists[0]): continue
            if base == 'a': ohe[i, j, 0] = 1
            if base == 'g': ohe[i, j, 1] = 1
            if base == 'c': ohe[i, j, 2] = 1
            if base == 't': ohe[i, j, 3] = 1
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
    for index in range(output_length):
        i = index // batch_size
        j = index % batch_size
        batched_data[i][j] = data[index]

    return batched_data


def train_val_test_split(data, train=0.7, val=0.2, test=0.1):
    length = len(data)
    return data[:train*length], \
           data[train*length:(train + val)*length], \
           data[(train + val)*length:(train + val + test)*length]


def load_data(seqs_path='seqs.npy', grna_path='grna.npy'):
    debug_print(['loading preprocessed data'])
    seqs, grna = np.load(seqs_path), np.load(grna_path)
    debug_print([
        'data shape:', 
        '\n                  DNA:  ', seqs.shape, 
        '\n                  gRNA: ', grna.shape])
    return seqs, grna


def get_train_test(df, length=1e4):
    debug_print(['loading genomic data from', GENOME_SEQUENCES_PATH])
    for record in SeqIO.parse(GENOME_SEQUENCES_PATH, 'fasta'):
        GENOME_SEQUENCES[record.id.split('.')[0]] = record.seq

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
            seq = fetch_genome_sequence(chromosome, start, end).lower()
            rna = row[0].lower()
            if set(list(seq)).union(bases) == set(list(seq)) and \
               set(list(rna)).union(bases) == set(list(rna)):
                seqs_list.append(seq)
                grna_list.append(rna)
        except:
            print(f'chromosome {chromosome} not found in the genome file.')
    grna = ohe_bases(grna_list)
    seqs = ohe_bases(seqs_list)
    debug_print([
        'data shape:', 
        '\n                  DNA:  ', seqs.shape, 
        '\n                  gRNA: ', grna.shape])

    return seqs, grna


def extract_data(path=DATA_PATH):
    debug_print(['loading gRNA data from', path])
    df = pd.read_csv(path)
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    return shuffled_df