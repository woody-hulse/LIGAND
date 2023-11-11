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

from GAN import *


def compute_baselines(models, X, Y):
    for model in models:
        baseline_pred = model.predict(X)
        baseline_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(Y, baseline_pred))
        debug_print([model.name, 'loss:', baseline_loss.numpy()])


def train(model, X, Y, epochs, batch_size=16, validation_split=0.2):
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam())
    model.build(X.shape)
    print(model.summary())
    model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

def train_multiproc(model, X, Y, epochs, batch_size=16, validation_split=0.2):
    # doesn't really work
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    debug_print(['number of devices:', strategy.num_replicas_in_sync])

    with strategy.scope():
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam())

    mirrored_X = tf.convert_to_tensor(X)
    mirrored_Y = tf.convert_to_tensor(Y)
    mirrored_dataset = tf.data.Dataset.from_tensor_slices((mirrored_X, mirrored_Y)).batch(batch_size)
    dist_dataset = strategy.experimental_distribute_dataset(mirrored_dataset)

    def train_step(inputs):
        X_batch, Y_batch = inputs

        with tf.GradientTape() as tape:
            predictions = model(X_batch, training=True)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(Y_batch, predictions))

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer = model.optimizer
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for inputs in dist_dataset:
            # Ensure the loss is initialized as a Tensor
            loss = strategy.run(train_step, args=(inputs,))
            total_loss += strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
            num_batches += 1

        # Calculate average training loss for this epoch
        average_loss = total_loss / num_batches

        print('Epoch {}, Loss: {:.4f}'.format(epoch, average_loss))

def main(load_data=False):
    if load_data:
        seqs, grna = preprocessing.load_data()
    else:
        df = preprocessing.extract_data()
        seqs, grna = preprocessing.get_train_test(df, 1e4)
        debug_print(['saving preprocessed data'])
        np.save('seqs.npy', seqs)
        np.save('grna.npy', grna)


    batch_size = 8
    batched_seqs = preprocessing.batch_data(seqs, batch_size)
    batched_grna = preprocessing.batch_data(grna, batch_size)

    # seqs_train, seqs_val, seqs_test = preprocessing.train_val_test_split(seqs)
    # grna_train, grna_val, grna_test = preprocessing.train_val_test_split(grna)

    compute_baselines([
        GuessBaseline(grna),
        MeanBaseline(grna)
    ], seqs, grna)

    model1 = ActorTransformer1(seqs.shape[1:], grna.shape[1:], num_transformers=4, hidden_size=32)
    model2 = ActorConvDeconv(seqs.shape[1:], grna.shape[1:])
    # model3 = tfm.nlp.models.TransformerDecoder(num_attention_heads=1)
    
    # train(model2, seqs, grna, 100)
    # train_multiproc(model2, seqs, grna, 100)
    
    gan = TestGAN(seqs.shape[1:], grna.shape[1:])
    gan.train(batched_seqs, batched_grna, 100)
    
    
    


if __name__ == '__main__':
    os.system('clear')

    main(True)