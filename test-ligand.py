import os

import math
import random
import numpy as np
import pandas as pd

import tensorflow as tf
# import tensorflow_models as tfm

import preprocessing
from preprocessing import debug_print
from models import ActorTransformer1, ActorConvDeconv
from models import GuessBaseline, MeanBaseline


def compute_baselines(models, X, Y):
    for model in models:
        baseline_pred = model.predict(X)
        baseline_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(Y, baseline_pred))
        debug_print([model.name, 'loss:', baseline_loss.numpy()])


def train(model, X, Y, epochs, batch_size=16, validation_split=0.2):
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam())
    model.build(X.shape + (1,))
    print(model.summary())
    model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)


def main(load_data=False):
    if load_data:
        debug_print(['loading preprocessed data'])
        seqs = np.load('seqs.npy')
        grna = np.load('grna.npy')
    else:
        df = preprocessing.extract_data()
        seqs, grna = preprocessing.get_train_test(df, 1e4)
        debug_print(['saving preprocessed data'])
        np.save('seqs.npy', seqs)
        np.save('grna.npy', grna)


    batch_size = 8
    # batched_seqs = batch_data(seqs, batch_size)
    # batched_grna = batch_data(grna, batch_size)

    # seqs_train, seqs_val, seqs_test = preprocessing.train_val_test_split(seqs)
    # grna_train, grna_val, grna_test = preprocessing.train_val_test_split(grna)

    compute_baselines([
        GuessBaseline(grna),
        MeanBaseline(grna)
    ], seqs, grna)

    model1 = ActorTransformer1(seqs.shape[1:], grna.shape[1:], num_transformers=4, hidden_size=32)
    model2 = ActorConvDeconv(seqs.shape[1:], grna.shape[1:])
    # model3 = tfm.nlp.models.TransformerDecoder(num_attention_heads=1)

    train(model2, seqs, grna, 100)
    


if __name__ == '__main__':
    os.system('clear')
    main(False)