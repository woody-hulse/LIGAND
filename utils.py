import datetime
import numpy as np

def debug_print(statements=[], end='\n'):
    ct = datetime.datetime.now()
    print('[', str(ct)[:19], '] ', sep='', end='')
    for statement in statements:
        print(statement, end=' ')
    print(end=end)

def ohe_base(base):
    base = base.lower()
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

def str_base(ohe):
    ohe = ohe[:4]
    if np.argmax(ohe) == 0: return 'a'
    if np.argmax(ohe) == 1: return 'g'
    if np.argmax(ohe) == 2: return 'c'
    if np.argmax(ohe) == 3: return 't'   

def str_bases(ohe):
    bases = ''
    for i in range(len(ohe)):
        bases += str_base(ohe[i])
    
    return bases

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

def prediction_to_sequence(prediction):
    seq = ''
    for base in prediction:
        nucleotide = np.argmax(base)
        if nucleotide == 0: seq += 'A'
        elif nucleotide == 1: seq += 'C'
        elif nucleotide == 2: seq += 'G'
        elif nucleotide == 3: seq += 'T'
    return seq

def ohe_dna_to_sequence(ohe_dna):
    seq = ''

    # truncate off epigenomic signals
    ohe_dna = ohe_dna[:, :4]

    for base in ohe_dna:
        nucleotide = np.argmax(base)
        if nucleotide == 0: seq += 'A'
        elif nucleotide == 1: seq += 'C'
        elif nucleotide == 2: seq += 'G'
        elif nucleotide == 3: seq += 'T'
    return seq

