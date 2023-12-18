import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

def train(models, X, Y, epochs, batch_size=64, validation_split=0.2, graph=True, summary=True, loss='categorical_crossentropy'):
    debug_print(['training model'])

    for model in models:
        model.compile(loss=loss, optimizer=tf.keras.optimizers.legacy.Adam(), metrics=['accuracy'])
        model(X)
        if summary: model.summary()
        model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    
    if graph:
        for model in models:
            val_loss = model.history.history['val_loss']
            plt.plot(val_loss, label=model.name + ' validation loss')
        plt.ylabel('categorical crossentropy loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
        
        for model in models:
            accuracy = model.history.history['accuracy']
            plt.plot(accuracy, label=model.name + ' accuracy')
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
    ## Preprocessing
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

    ## Baselines
    compute_baselines([
        GuessBaseline(grna),
        MeanBaseline(grna),
        CenterBaseline(grna),
        PairBaseline()], seqs, grna)

    ## Models
    # Change GAN below. GANS can be found in GAN.py. Models can be found in models.py.
    gan = Trans_Conv_GAN2(seqs.shape[1:], grna.shape[1:])

    train([gan.generator], seqs, grna, epochs=0, graph=False, summary=False)
    train([gan.discriminator], [seqs, grna], np.ones(len(seqs)), epochs=0, graph=False, summary=False)

    ## Training
    gan.train(batched_seqs_train, 
              batched_grna_train, 
              epochs=3, 
              validation_data=(seqs_val, grna_val), 
              print_interval=1, summary=True, plot=False,
              save=False, load=True)

    # save_roc(seqs_test, grna_test, gan.generator, file=f'models/{gan.name}/roc.csv')
        
    ## Analysis
    rnas, chromosomes, starts, ends = preprocessing.get_activity_tests(df, 512, load_data)

    print(rnas.shape)

    validation_activity_map(gan)
    validate_against_efficacies(gan)
    
    activity_test(
            gan=gan,
            rnas=rnas,
            chromosomes=chromosomes,
            starts=starts,
            ends=ends,
            a=50,
            num_seqs=2)
    
    diffs = []
    for base in ['a', 'g', 'c', 't']:
        diff = perturbation_analysis(
            gan=gan,
            rnas=rnas,
            chromosomes=chromosomes,
            starts=starts,
            ends=ends,
            base=base,
            num_seqs=30,
            a=50,
            plot=False
        )
        diff = np.array(diff).mean(axis=0)
        diffs.append(diff)
    
    x = np.arange(0,4)
    diffs = np.array(diffs).T
    plt.imshow(diffs, cmap='Blues', origin='lower', aspect='auto', extent=(0, 4, 0, 20))
    plt.colorbar(label='Percent Difference')
    plt.xlabel('Replacement Base')
    plt.ylabel(f'gRNA Perturbation Index')
    plt.yticks(np.arange(0, 20))
    plt.xticks(x,['a', 'g', 'c', 't'])
    plt.grid(axis='y', linestyle='solid', alpha=0.7)
    plt.title(f'Percent Difference in Activity at Target for Replacement Bases')
    plt.show()

    '''

    activity_scores_avg = activity_test(
        gan=gan,
        rnas=rnas,
        chromosomes=chromosomes,
        starts=starts,
        ends=ends,
        a=500,
        num_seqs=256,
        plot=False)
    
    plt.figure(figsize=(8, 4))
    plt.plot(activity_scores_avg)
    plt.ylabel('average predicted activity')
    plt.xlabel('genomic position')
    plt.axvline(x=500, color='orange', linestyle='dotted', label='bind site')
    plt.show()
    
    best_five = candidate_grna_range(gan, 
                                     chromosome=chromosomes[0], 
                                     start=starts[0], 
                                     end=ends[0], 
                                     a=250,
                                     num_seqs=5)
    deviation_from_complement_dna(gan.generator, seqs_test)

    perturbation_map(
        gan=gan,
        rnas=rnas,
        chromosomes=chromosomes,
        starts=starts,
        ends=ends,
        view_length=23,
        num_seqs=4
    )
    
    complement_activity_test(
        gan=gan,
        chromosome=chromosomes[0],
        start=starts[0],
        end=ends[0],
        a=50)
    
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