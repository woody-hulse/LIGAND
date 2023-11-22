import tensorflow as tf
import numpy as np

from tqdm import tqdm

import preprocessing
from models import *


class TestGenerator(tf.keras.Model):
    def __init__(self, output_shape, name='test_generator', **kwargs):
        super().__init__(name=name, **kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(output_shape[0] * 4, activation='relu')
        self.reshape = tf.keras.layers.Reshape((output_shape[0], 4))
        self.out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation='softmax'))

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.reshape(x)
        x = self.out(x)
        
        return x


class TestDiscriminator(tf.keras.Model):
    def __init__(self, name='test_discriminator', **kwargs):
        super().__init__(name=name, **kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.denseGRNA1 = tf.keras.layers.Dense(64, activation='relu')
        self.denseGRNA2 = tf.keras.layers.Dense(32, activation='relu')
        self.denseSEQS1 = tf.keras.layers.Dense(64, activation='relu')
        self.denseSEQS2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x_seqs, x_grna = x

        x_grna = self.flatten(x_grna)
        x_grna = self.denseGRNA1(x_grna)
        x_grna = self.denseGRNA2(x_grna)

        x_seqs = self.flatten(x_seqs)
        x_seqs = self.denseSEQS1(x_seqs)
        x_seqs = self.denseSEQS2(x_seqs)

        x = tf.concat([x_grna, x_seqs], axis=1)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


class TestGAN(tf.keras.Model):
    def __init__(self, input_shape, output_shape, name='test_gan', **kwargs):
        super().__init__(name=name, **kwargs)
        self.shape = input_shape
        self.generator = TestGenerator(output_shape)
        self.generator.build((1,) + input_shape)
        self.discriminator = TestDiscriminator()
        test_data = [np.zeros((1,) + input_shape), np.zeros((1,) + output_shape)]
        self.discriminator(test_data)
    
    def train(self, X, Y, epochs, batch_size=8, learning_rate=0.01):
        generator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)
        discriminator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)

        cross_entropy = tf.keras.losses.BinaryCrossentropy()

        mean_probs = np.full(Y[0].shape, .25)
        
        confidence = 0
        for epoch in range(1, epochs + 1):
            example_probs = None
            example_labels = None
            for i in tqdm(range(X.shape[0]), desc='epoch {} : confidence {:.2f} '.format(str(epoch).zfill(4), confidence)):
                noise = tf.random.normal((batch_size,) + self.shape)
                generated_probs = self.generator(X[i])
                example_probs = generated_probs[0]
                example_labels = Y[i][0]
                confidence = tf.keras.losses.MeanSquaredError()(generated_probs, mean_probs) * 100
                generated_sequences = tf.one_hot(tf.math.argmax(generated_probs, axis=2), 4, axis=2)
                # generated_sequences = self.generator(generated_probs)
                real_labels = tf.ones((batch_size, 1))
                fake_labels = tf.zeros((batch_size, 1))

                with tf.GradientTape() as disc_tape:
                    d_loss_real = cross_entropy(real_labels, self.discriminator([X[i], Y[i]]))
                    d_loss_fake = cross_entropy(fake_labels, self.discriminator([X[i], generated_sequences]))
                    d_loss = d_loss_real + d_loss_fake

                gradients_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
                discriminator_optimizer.apply_gradients(zip(gradients_discriminator, self.discriminator.trainable_variables))

                with tf.GradientTape() as gen_tape:
                    generated_probs = self.generator(X[i])
                    generated_sequences = tf.one_hot(tf.math.argmax(generated_probs, axis=2), 4, axis=2)
                    g_loss = cross_entropy(fake_labels, self.discriminator([X[i], generated_probs]))

                gradients_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
                generator_optimizer.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))

            if epoch % 10 == 0:
                preprocessing.debug_print(['epoch', epoch, 
                                           ':\n           generator loss :', g_loss.numpy(),
                                           ' \n       discriminator loss :', d_loss.numpy()])
                print(example_probs)
                print(example_labels)




