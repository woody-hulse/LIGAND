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


def compute_baselines(models, X, Y):
    for model in models:
        baseline_pred = model.predict(X)
        baseline_loss = tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(Y, baseline_pred))
        debug_print([model.name, 'loss:', baseline_loss.numpy()])


def train(model, X, Y, epochs, batch_size=64, validation_split=0.2, graph=True, summary=True, loss='categorical_crossentropy'):
    debug_print(['training model'])

    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
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
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam())

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
        
        

def activity_test(discriminator, rna, chromosome, start, end, view_length=23, bind_site=-1, plot=True):
    
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    if bind_site == -1: bind_site = (start + end) // 2
        
    ohe_rna = np.concatenate([preprocessing.ohe_base(base) for base in rna], axis=0)
    seq = preprocessing.fetch_genomic_sequence(chromosome, start, end).lower()
    ohe_seq = np.concatenate([preprocessing.ohe_base(base) for base in seq], axis=0)
    epigenomic_signals = preprocessing.fetch_epigenomic_signals(chromosome, start, end)
    epigenomic_seq = np.concatenate([ohe_seq, epigenomic_signals], axis=1)
    
    activity_scores = []
    for i in range(end - start - view_length):
        if discriminator.name == 'conv_discriminator':
            activity_score = discriminator([
                np.expand_dims(epigenomic_seq[i:i+view_length], axis=0), 
                np.expand_dims(ohe_rna, axis=0)
            ])
            activity_scores.append(activity_score[0][0])
        else:
            pass # finish
    activity_scores = np.array(activity_scores)
    moving_averages = moving_average(activity_scores, 23)
        
    if plot:
        x = np.arange(start + view_length, end - view_length + 1)
        plt.figure(figsize=(8, 3))
        plt.plot(x, moving_averages, label=rna)
        # plt.plot(x, activity_scores[11:-11], label='test')
        plt.axvline(x=bind_site, color='orange', linestyle='dotted', label='bind site')
        plt.title('GRNA activity over chromosome ' + chromosome)
        plt.xlabel('genomic position')
        plt.ylabel('predicted activity (0-1)')
        plt.legend()
        plt.show()
    
    return activity_scores, moving_averages

def perturbation_analysis(discriminator, chromosome, start, end, bind_site, rna, perturbation_length, base='N'):
    RNA = rna
    heatmap = np.zeros((len(RNA) - perturbation_length + 1,777))
    _,original = activity_test(discriminator=discriminator,rna=RNA,chromosome=chromosome,start=start,end=end,bind_site=bind_site, plot=False)
    middle = len(original)//2
    orig_target_mean = np.mean(original[middle-20:middle+50])
    perturbed_target_means = np.zeros(len(RNA) - perturbation_length + 1)
    
    for i in range(0, len(RNA)-perturbation_length + 1):
        perturbed_grna = list(RNA)
        perturbed_grna[i:i + perturbation_length] = base * perturbation_length
        perturbed_grna = ''.join(perturbed_grna)
        _, averages = activity_test(discriminator=discriminator,rna=perturbed_grna,chromosome=chromosome,start=start,end=end,bind_site=bind_site, plot=False)
        heatmap[i,:] = averages
        perturbed_target_means[i] = (abs(np.mean(averages[middle-20:middle+50]) - orig_target_mean)/orig_target_mean)
    
    plt.figure(1)
    x = np.arange(start + 23, end - 23 + 1)
    plt.imshow(heatmap, cmap='inferno', origin='lower', aspect='auto', extent=(min(x), max(x), 0, len(RNA)- perturbation_length + 1))
    plt.colorbar(label='Activity Score')
    plt.xlabel('DNA Position')
    plt.ylabel(f'gRNA Perturbation Index (Length {perturbation_length}, Base {base})')
    plt.yticks(np.arange(0, len(RNA)-perturbation_length + 1))
    plt.grid(axis='y', linestyle='solid', alpha=0.7)
    plt.title('Perturbation Analysis Heatmap')
    plt.show()

    plt.figure(2)
    x = np.arange(0, len(RNA)- perturbation_length + 1)
    plt.bar(x, perturbed_target_means, color='maroon', width=.4)
    plt.xlabel(f'gRNA Perturbation Index (Length {perturbation_length}, Base {base})')
    plt.ylabel('Percent Difference In Activity Score at Target vs Original')
    plt.title('Perturbation Index vs Effect on Target Bind Activity Percent Change')
    plt.show()

def main(load_data=False):
    if load_data:
        seqs, grna = preprocessing.load_data()
    else:
        df = preprocessing.extract_data()
        seqs, grna = preprocessing.get_train_test(df, 1e4)
        debug_print(['saving preprocessed data'])
        np.save('data/seqs.npy', seqs)
        np.save('data/grna.npy', grna)


    batch_size = 32
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
    gan = TestGAN(seqs.shape[1:], grna.shape[1:])
    train(gan.generator, seqs, grna, epochs=0, graph=False, summary=False)
    train(gan.discriminator, [seqs, grna], np.ones(len(seqs)), epochs=0, graph=False, summary=False)
    gan.train(batched_seqs_train, 
              batched_grna_train, 
              epochs=50, 
              validation_data=(seqs_val, grna_val), 
              print_interval=1, summary=True, plot=True,
              save=False, load=False)
    
    # discriminator sliding window
    preprocessing.read_genome()
    activity_test(
        discriminator=gan.discriminator,
        rna='GAATGGGAGAGAATATCACT',
        chromosome='18',
        start=7080001 - 400,
        end=7080023 + 400,
        bind_site=7080001)
    
    for base in ['A', 'G', 'C', 'T']:
        for perturb_len in [1,3,5]:
            perturbation_analysis(discriminator=gan.discriminator,
                rna='GAATGGGAGAGAATATCACT',
                chromosome='18',
                start=7080001 - 400,
                end=7080023 + 400,
                bind_site=7080001,
                perturbation_length=perturb_len,
                base=base)
    

if __name__ == '__main__':
    os.system('clear')

    main(False)