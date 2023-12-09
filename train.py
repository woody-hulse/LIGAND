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



def generate_candidate_grna(gan, chromosome, start, end, a=400, view_length=23, num_seqs=4, plot=True):
    
    chromosomes = [chromosome for _ in range(num_seqs)]
    starts = [start for _ in range(num_seqs)]
    ends = [end for _ in range(num_seqs)]
    
    if plot:
        fig, axis = plt.subplots(num_seqs, 1, figsize=(8, num_seqs * 2))
        axis[0].set_title('GRNA activity')
        
    X = np.zeros((num_seqs, view_length, 8))
    seq = None
    for n in range(num_seqs):
        seq = preprocessing.fetch_genomic_sequence(chromosome, start, end).lower()
        ohe_seq = np.concatenate([preprocessing.ohe_base(base) for base in seq], axis=0)
        epigenomic_signals = preprocessing.fetch_epigenomic_signals(chromosome, start, end)
        epigenomic_seq = np.concatenate([ohe_seq, epigenomic_signals], axis=1)
        
        X[n] = epigenomic_seq
        
    candidate_grna = gan.generate(X)
    
    debug_print(['generating candidate grna for', seq, ':'])
    for grna in candidate_grna:
        debug_print(['     ', preprocessing.str_bases(grna)])
    
    activity_test(gan, candidate_grna, chromosomes, starts, ends, a, view_length, plot, num_seqs, True)



def activity_test(gan, rnas, chromosomes, starts, ends, a=400, view_length=23, plot=True, num_seqs=None, test_rna=False):
    if not num_seqs: num_seqs = len(rnas)
    
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    if plot:
        fig, axis = plt.subplots(num_seqs, 1, figsize=(8, num_seqs * 2))
        axis[0].set_title('GRNA activity')
    
    skipped = 0
    skip = []
    
    all_activity_scores = []
    
    X_gen = np.zeros((len(rnas), view_length, 8))
    X = np.zeros((len(rnas), view_length + 2 * a, 8))
    
    if test_rna:
        real_Yi = np.array(rnas)
    else:
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
    for n, (rna, chromosome, start, end) in enumerate(zip(rnas, chromosomes, starts, ends)):
        if n >= num_seqs + skipped: continue
        if n in skip:
            skipped += 1
            continue
        preprocessing.debug_print(['running activity test on', preprocessing.str_bases(rna)])
        bind_site = start
        start -= a
        end += a
        
        activity_scores = []
        step = 1
        for i in range(0, end - start - view_length, step):
            if gan.discriminator.name == 'conv_discriminator':
                activity_score = gan.discriminator([
                    np.expand_dims(X[n][i:i+view_length], axis=0), 
                    np.expand_dims(real_Yi[n], axis=0)
                ])
                for _ in range(step):
                    activity_scores.append(activity_score[0][0].numpy())
            else:
                pass # finish
        activity_scores = np.array(activity_scores)
        moving_averages = activity_scores[23:]# moving_average(activity_scores, 100)
        all_activity_scores.append(activity_scores)
            
        if plot:
            x = np.arange(start + view_length, end - view_length + 4)[:len(moving_averages)]
            axis[axis_index].plot(x, moving_averages, label=preprocessing.str_bases(rna))
            axis[axis_index].axvline(x=bind_site, color='orange', linestyle='dotted', label='bind site')
            axis[axis_index].legend()
        
        axis_index += 1
    
    if plot:
        plt.xlabel('genomic position')
        plt.ylabel('predicted activity')
        plt.show()
    
    return all_activity_scores


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
            heatmap[i, :] = np.array(perturbed_activity_scores)[view_length - 1:]
        
        axis_index += 1
    
        x = np.arange(start + 23, end - 23 + 1)
        plt.imshow(heatmap, cmap='inferno', origin='lower', aspect='auto', extent=(min(x), max(x), 0, len(rna)))
        plt.colorbar(label='Activity Score')
        plt.xlabel('DNA Position')
        plt.ylabel(f'gRNA Perturbation Index (Length {1}, Base {base})')
        plt.yticks(np.arange(0, len(rna)))
        plt.grid(axis='y', linestyle='solid', alpha=0.7)
        plt.title('Perturbation Analysis Heatmap')
        plt.show()
        
    

'''
def perturbation_analysis(gan, chromosome, start, end, bind_site, rna, perturbation_length, base='N'):
    RNA = rna
    heatmap = np.zeros((len(RNA) - perturbation_length + 1, 777))
    _,original = activity_test(gan=gan,rna=RNA,chromosome=chromosome,start=start,end=end,bind_site=bind_site, plot=False)
    middle = len(original)//2
    orig_target_mean = np.mean(original[middle-20:middle+50])
    perturbed_target_means = np.zeros(len(RNA) - perturbation_length + 1)
    
    for i in range(0, len(RNA)-perturbation_length + 1):
        perturbed_grna = list(RNA)
        perturbed_grna[i:i + perturbation_length] = base * perturbation_length
        perturbed_grna = ''.join(perturbed_grna)
        _, averages = activity_test(gan=gan,rna=perturbed_grna,chromosome=chromosome,start=start,end=end,bind_site=bind_site, plot=False)
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
'''


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
    gan = TestGAN(seqs.shape[1:], grna.shape[1:])
    train(gan.generator, seqs, grna, epochs=0, graph=False, summary=False)
    train(gan.discriminator, [seqs, grna], np.ones(len(seqs)), epochs=0, graph=False, summary=False)
    gan.train(batched_seqs_train, 
              batched_grna_train, 
              epochs=0, 
              validation_data=(seqs_val, grna_val), 
              print_interval=1, summary=True, plot=False,
              save=False, load=True)
    
    # discriminator sliding window
    rnas, chromosomes, starts, ends = preprocessing.get_activity_tests(df, batch_size, load_data)
    
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
    
    generate_candidate_grna(
        gan=gan, 
        chromosome=chromosomes[0], 
        start=starts[0], 
        end=ends[0], 
        a=400,
        num_seqs=4,
        plot=True)


if __name__ == '__main__':
    os.system('clear')

    main(True)