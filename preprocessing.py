import csv
from tqdm import tqdm
import random
import numpy as np
import pandas as pd

from Bio import SeqIO
import pyBigWig

from utils import *

H3K4ME3_PATH = "data/h3k4me.bigWig" # https://drive.google.com/drive/u/0/folders/1nycERZkXh5Qiyy8HQE3Hk0pOez_UFECw
RRBS_PATH = "data/methyl.bigBed"
DNASE_PATH = "data/dnase.bigWig"
CTCF_PATH = "data/ctcf.bigWig"

REFERENCE_GENOME_PATH = 'data/GCF_000001405.26_GRCh38_genomic.fna' # https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.26/
GRNA_PATH = 'data/hg_guide_info.csv' # https://www.ncbi.nlm.nih.gov/genome/guide/human/

GENOME_SEQUENCES = {}
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

def get_activity_tests(df, num_seqs, read=True):
    if (read): read_genome()
    
    rnas = []
    chromosomes = []
    starts = []
    ends = []
    
    exclude = set(['0X', '0Y', '0M'])
    bases = set(['a', 'c', 'g', 't'])
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    for index, row in tqdm(shuffled_df.iterrows()):
        if len(rnas) >= num_seqs: break
        chromosome = row.iloc[3][3:].zfill(2)
        if chromosome in exclude: continue
        if row.iloc[0][0] == 'N': continue
        start, end = row.iloc[4], row.iloc[5]
        
        fetch_genomic_sequence(chromosome, start, end).lower()
        rna = row.iloc[0].lower()
            
        rnas.append(np.concatenate([ohe_base(base) for base in rna], axis=0))
        chromosomes.append(chromosome)
        starts.append(start)
        ends.append(end)
    
    return np.array(rnas), chromosomes, starts, ends

def get_train_test(df, length=1e4):
    read_genome()

    grna_list = []
    seqs_list = []
    exclude = set(['0X', '0Y', '0M'])
    bases = set(['a', 'c', 'g', 't'])
    debug_print(['locating corresponding genome sequences'])
    for index, row in tqdm(df.iterrows()):
        if len(grna_list) >= length: break
        chromosome = row.iloc[3][3:].zfill(2)
        if chromosome in exclude: continue
        if row.iloc[0][0] == 'N': continue
        start, end = row.iloc[4], row.iloc[5]
        try:
            seq = fetch_genomic_sequence(chromosome, start, end).lower()
            rna = row.iloc[0].lower()
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

# Efficacies
EFFICACY_PATHS= {
    'hct':'data/ontar/hct116.csv',
    'hek':'data/ontar/hek293t.csv',
    'hela':'data/ontar/hela.csv',
    'hl60':'data/ontar/hl60.csv',
    'offtar_off':'data/offtar_off.csv',
}
EFFICACY_MAP = {}
gRNA = []

# Not using cell type right now
def populate_efficacy_map():
    for cell_type in EFFICACY_PATHS:
        sgRNA_map = {}
        path = EFFICACY_PATHS[cell_type]
        with open(path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                chromosome = row['Chromosome']
                start = int(row['Start'])
                end = int(row['End'])
                sgRNA = ''
                efficacy = 0
                if cell_type == 'offtar_off':
                    sgRNA = row['OT']
                    efficacy = float(row['Cleavage Frequency'])
                else:
                    sgRNA = row['sgRNA']
                    efficacy = float(row['Normalized efficacy'])
                gRNA.append(sgRNA)
                key_tuple = (sgRNA, chromosome, start, end)
                sgRNA_map[key_tuple] = efficacy
                EFFICACY_MAP[key_tuple] = efficacy
    return EFFICACY_MAP

def get_efficacy(sgRNA, chromosome, start, end):
    key_tuple = (sgRNA, chromosome, start, end)
    return EFFICACY_MAP[key_tuple]

def get_random_efficacy(targ_GRNA, chromosome, start, end):
    key = (random.choice(gRNA), chromosome, start, end)
    count = 0
    while key not in EFFICACY_MAP or key[0] == targ_GRNA:
        key = (random.choice(gRNA), chromosome, start, end)
        count +=1
    return key[0]

