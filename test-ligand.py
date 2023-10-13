import csv
import os
from tqdm import tqdm
import datetime

import math
import random
import numpy as np
import pandas as pd

import tensorflow as tf

from Bio import SeqIO
genome_sequences = {}
for record in SeqIO.parse('GRCh37_latest_genomic.fna', 'fasta'):
    genome_sequences[record.id.split('.')[0]] = record.seq


def debug_print(statements=[], end='\n'):
    ct = datetime.datetime.now()
    print('[', str(ct)[:19], '] ', sep='', end='')
    for statement in statements:
        print(statement, end=' ')
    print(end=end)


def read_genome(df, path='GRCh37_latest_genomic.fna'):
    tm = SeqIO.parse(path, 'fasta')
    debug_print(['dna sequence imported from', path])


def fetch_genome_sequence(chromosome, start, end):
    chromosome_id = f'NC_0000{chromosome}'
    genome_sequence = genome_sequences.get(chromosome_id)
    if genome_sequence:
        return str(genome_sequence[start:end]) # start - 1
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


def get_train_test(df, length=10000):
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
    seqs = tokenize_bases(seqs_list)
    debug_print([
        'output data shape:', 
        '\n                      DNA:  ', seqs.shape, 
        '\n                      gRNA: ', grna.shape])

    return seqs, grna


def extract_data(path='hg_guide_info.csv'):
    debug_print(['importing gRNA data from', path])
    df = pd.read_csv(path)

    return df


class Transformer(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_size, ff_dim, dropout=0.1, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_size
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation='relu'),
            tf.keras.layers.Dense(input_shape[-1])
        ])
        self.dropout1 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=True):
        attn_output = self.attention(query=inputs, value=inputs, attention_mask=None, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    

class Actor(tf.keras.Model):
    def __init__(self, input_shape, output_shape, num_transformers=3, hidden_size=32):
        super().__init__()

        embedding_dim = 4
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=4, output_dim=embedding_dim, input_length=input_shape[0])
        self.transformers = tf.keras.Sequential([Transformer(8, 8, hidden_size) for _ in range(num_transformers)])
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_shape[0] * output_shape[1], activation='relu')
        self.reshape = tf.keras.layers.Reshape(output_shape)
        self.dense3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation='softmax'))

    def call(self, x):
        x = self.embedding_layer(x)
        x = self.transformers(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)
        x = self.dense3(x)

        return x

    def predict(self, x):
        return self.call(x)


def main(load_data=False):
    if load_data:
        seqs = np.load('seqs.npy')
        grna = np.load('grna.npy')
    else:
        df = extract_data()
        seqs, grna = get_train_test(df)
        np.save('seqs.npy', seqs)
        np.save('grna.npy', grna)

    batch_size = 8
    # batched_seqs = batch_data(seqs, batch_size)
    # batched_grna = batch_data(grna, batch_size)

    model = Actor(seqs.shape[1:], grna.shape[1:], num_transformers=4, hidden_size=32)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam())
    model.build(seqs.shape)
    print(model.summary())
    model.fit(seqs, grna, batch_size=batch_size, epochs=100)

    print(model.predict(np.array([seqs[10000]])))
    print(grna[10000])


if __name__ == '__main__':
    os.system('clear')
    main()