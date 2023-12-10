import os

import math
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
# import tensorflow_models as tfm

import preprocessing
from preprocessing import debug_print

from models import *
from GAN import *
from analysis import *


def compute_baselines(models, X, Y):
    for model in models:
        baseline_pred = model.predict(X)
        baseline_loss = tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(Y, baseline_pred))
        debug_print([model.name, 'loss:', baseline_loss.numpy()])

def train(model, X, Y, epochs, batch_size=64, validation_split=0.2, graph=True, summary=True, loss='categorical_crossentropy'):
    debug_print(['training model'])

    model.compile(loss=loss, optimizer=tf.keras.optimizers.legacy.Adam(), metrics=['accuracy'])
    model(X)
    if summary: model.summary()
    model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    
    if graph:
        loss = model.history.history['loss']
        val_loss = model.history.history['val_loss']
        plt.plot(loss, label='training')
        plt.plot(val_loss, label='validation')
        plt.ylabel('categorical crossentropy loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

def train_multiproc(model, X, Y, epochs, batch_size=16, validation_split=0.2):
    # doesn't really work
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    debug_print(['number of devices:', strategy.num_replicas_in_sync])

    with strategy.scope():
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam())

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

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for inputs in dist_dataset:
            loss = strategy.run(train_step, args=(inputs,))
            total_loss += strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
            num_batches += 1

        average_loss = total_loss / num_batches

        debug_print(['epoch', epoch, 'loss :', average_loss])


def main(load_data=False):
    if load_data:
        df = preprocessing.extract_data()
        seqs, grna = preprocessing.load_data()
    else:
        df = preprocessing.extract_data()
        seqs, grna = preprocessing.get_train_test(df, 1e4)
        debug_print(['saving preprocessed data'])
        np.save('data/seqs.npy', seqs)
        np.save('data/grna.npy', grna)


    batch_size = 128
    batched_seqs = preprocessing.batch_data(seqs, batch_size)
    batched_grna = preprocessing.batch_data(grna, batch_size)
    
    batched_seqs_train, batched_seqs_val, batched_seqs_test = preprocessing.train_val_test_split(batched_seqs)
    batched_grna_train, batched_grna_val, batched_grna_test = preprocessing.train_val_test_split(batched_grna)

    seqs_train, seqs_val, seqs_test = preprocessing.train_val_test_split(seqs)
    grna_train, grna_val, grna_test = preprocessing.train_val_test_split(grna)

    compute_baselines([
        GuessBaseline(grna),
        MeanBaseline(grna),
        CenterBaseline(grna),
        PairBaseline()
    ], seqs, grna)

    # actor model training
    model1 = ActorTransformer1(seqs.shape[1:], grna.shape[1:], num_transformers=4, hidden_size=32)
    model2 = ActorConvDeconv(seqs.shape[1:], grna.shape[1:])
    model3 = ActorDense(seqs.shape[1:], grna.shape[1:])
    # model4 = tfm.nlp.models.TransformerDecoder(num_attention_heads=1)
    # train(model2, seqs, grna, epochs=100)
    # train_multiproc(model2, seqs, grna, 100)
    
    # discriminator model training
    # discriminator_seqs, discriminator_grna = preprocessing.get_discriminator_train_test(seqs, grna)
    # discriminator = TestDiscriminator()
    # train(discriminator, discriminator_seqs, discriminator_grna, epochs=100, loss='binary_crossentropy')
    
    # gan model training
    gan = Trans_Conv_GAN2(seqs.shape[1:], grna.shape[1:])

    train(gan.generator, seqs, grna, epochs=0, graph=False, summary=False)
    train(gan.discriminator, [seqs, grna], np.ones(len(seqs)), epochs=0, graph=False, summary=False)
    gan.train(batched_seqs_train, 
              batched_grna_train, 
              epochs=50, 
              validation_data=(seqs_val, grna_val), 
              print_interval=1, summary=True, plot=False,
              save=True, load=True)
    
     # save metrics
    # save_roc(seqs_test, grna_test, gan.generator, file=f'models/{gan.name}/roc.csv')

    # discriminator sliding window
    rnas, chromosomes, starts, ends = preprocessing.get_activity_tests(df, batch_size, load_data)
    
    # deviation_from_complement_dna(gan.generator, seqs_test)


    complement_activity_test(
        gan=gan,
        chromosome=chromosomes[0],
        start=starts[0],
        end=ends[0],
        a=50)
    
    '''

    activity_test(
        gan=gan,
        rnas=rnas,
        chromosomes=chromosomes,
        starts=starts,
        ends=ends,
        a=50,
        num_seqs=4)
    
    for base in ['a', 'g', 'c', 't']:
        perturbation_analysis(
            gan=gan,
            rnas=rnas,
            chromosomes=chromosomes,
            starts=starts,
            ends=ends,
            base=base,
            num_seqs=4,
            a=50
        )
    
    for i in range(len(rnas)):
        generate_candidate_grna(
            gan=gan, 
            rna=rnas[i],
            chromosome=chromosomes[i], 
            start=starts[i], 
            end=ends[i], 
            a=50,
            num_seqs=6,
            plot=True)
    '''


if __name__ == '__main__':
    os.system('clear')

    main(True)