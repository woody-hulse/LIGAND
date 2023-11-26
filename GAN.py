import tensorflow as tf
import numpy as np

from tqdm import tqdm

import preprocessing
from preprocessing import debug_print
from models import *


def discriminator_loss(real_output, pred_output):
    real_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(pred_output), pred_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(pred_output):
    return tf.keras.losses.BinaryCrossentropy()(tf.ones_like(pred_output), pred_output)


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
        regularizer = tf.keras.regularizers.l2(0.01)
        self.flatten = tf.keras.layers.Flatten()
        self.denseGRNA1 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizer)
        self.denseGRNA2 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizer)
        self.denseSEQS1 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizer)
        self.denseSEQS2 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizer)
        self.dense1 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizer)
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizer)

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
        self.generator = ActorConvDeconv(input_shape, output_shape)
        self.generator.build((1,) + input_shape)
        self.discriminator = TestDiscriminator()
        test_data = [np.zeros((1,) + input_shape), np.zeros((1,) + output_shape)]
        self.discriminator(test_data)
    
    def train(self, X, Y, epochs, batch_size=8, learning_rate=0.01, print_interval=1):
        debug_print(['training GAN'])
        generator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)
        discriminator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)

        mean_probs = np.full(Y[0].shape, .25)
        
        confidence = 0
        for epoch in range(1, epochs + 1):
            example_probs = None
            example_labels = None

            gen_loss = None
            disc_loss = None
            for i in tqdm(range(X.shape[0]), desc='epoch {} : confidence {:.2f}'.format(str(epoch).zfill(4), confidence)):
                noise = tf.random.normal((batch_size,) + self.shape)
                generated_probs = self.generator(X[i])
                example_probs = generated_probs[0]
                example_labels = Y[i][0]
                confidence = tf.keras.losses.MeanSquaredError()(generated_probs, mean_probs) * 100
                generated_sequences = tf.one_hot(tf.math.argmax(generated_probs, axis=2), 4, axis=2)
                # generated_sequences = self.generator(generated_probs)

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    pred_Yi = self.generator(X[i])
                    pred = np.argmax(pred_Yi, axis=2)
                    real = np.argmax(Y[i], axis=2)
                    real_Yi = pred_Yi.numpy()

                    for m in range(Y[i].shape[0]):
                        for n in range(Y[i].shape[1]):
                            real_Yi[m][n][pred[m][n]], real_Yi[m][n][real[m][n]] = real_Yi[m][n][real[m][n]], real_Yi[m][n][pred[m][n]]

                    real_output = self.discriminator([X[i], real_Yi])
                    pred_output = self.discriminator([X[i], pred_Yi])
                    # pred_output2 = self.discriminator([X[i], tf.one_hot(tf.math.argmax(pred_Yi, axis=2), 4, axis=2)])

                    gen_loss = generator_loss(pred_output)
                    disc_loss = discriminator_loss(real_output, pred_output)
                    print(gen_loss.numpy(), disc_loss.numpy())
                    # print(pred_output.numpy(), real_output.numpy())

                gradients_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                generator_optimizer.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))

                gradients_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
                discriminator_optimizer.apply_gradients(zip(gradients_discriminator, self.discriminator.trainable_variables))

            if epoch % print_interval == 0:
                print(self.generator(X[0]))
                gen_real_loss, gen_accuracy = self.generator.evaluate(X[0], Y[0])

                preprocessing.debug_print(['epoch', epoch, 
                                           ':\n           generator GAN loss :', gen_loss.numpy(),
                                           ' \n               generator loss :', gen_real_loss, 
                                           ' \n           generator accuracy :', gen_accuracy,
                                           ' \n       discriminator GAN loss :', disc_loss.numpy()])




