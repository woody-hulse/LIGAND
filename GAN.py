import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

import preprocessing
from preprocessing import debug_print
from models import *


def discriminator_loss(real_output, pred_output):
    real_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(pred_output), pred_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(pred, real, pred_output):
    ratio = 0.5
    gen  = tf.keras.losses.CategoricalCrossentropy()(real, pred)
    disc = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(pred_output), pred_output)
    return gen * ratio + disc * (1 - ratio)


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
        regularizer = tf.keras.regularizers.l2(0.0001)
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


class ConvDiscriminator(tf.keras.Model):
    def __init__(self, input_shape=(23, 12, 1), name='conv_discriminator', **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, input_shape[2]), padding='valid', activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def preprocess_input(self, seqs, grna):
        output_shape = (seqs.shape[0], seqs.shape[1], grna.shape[2])
        pad_width = [(0, 0), (0, output_shape[1] - grna.shape[1]), (0, 0)]
        padded_grna = np.pad(grna, pad_width, mode='constant', constant_values=0)
        concat = np.concatenate([padded_grna, seqs], axis=2)
        x = concat[..., np.newaxis]
        return x
        

    def call(self, x):
        x = self.preprocess_input(*x)
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x


class TestGAN(tf.keras.Model):
    def __init__(self, input_shape, output_shape, name='test_gan', **kwargs):
        super().__init__(name=name, **kwargs)
        self.shape = input_shape
        self.generator = ActorConvDeconv(input_shape, output_shape)
        self.generator.build((1,) + input_shape)
        # self.discriminator = TestDiscriminator()
        self.discriminator = ConvDiscriminator()
        test_data = [np.zeros((1,) + input_shape), np.zeros((1,) + output_shape)]
        self.discriminator(test_data)
        
    
    def save_model(self):
        debug_print(['saving GAN'])
        self.generator.save('models/generator.keras')
        self.discriminator.save('models/discriminator.keras')
    
    
    def load_model(self):
        debug_print(['loading GAN'])
        self.generator = tf.keras.models.load_model('models/generator.keras')
        self.discriminator = tf.keras.models.load_model('models/discriminator.keras')
    
    
    def train(self, 
              X, Y, 
              epochs, 
              validation_data=(), 
              batch_size=8, 
              learning_rate=0.001, 
              print_interval=1, 
              summary=True, plot=True,
              save=True, load=False):
        
        if load:
            self.load_model()
        
        if summary:
            print()
            debug_print(['generator architecture:'])
            self.generator.summary()
            print()
            debug_print(['discriminator architecture:'])
            self.discriminator.summary()
            print()
        
        debug_print(['training GAN'])
        generator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)
        discriminator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)

        mean_probs = np.full(Y[0].shape, .25)
        
        gen_losses = []
        gen_real_losses = []
        gen_accuracies = []
        disc_losses = []
        disc_accuracies = []
        
        confidence = 0
        for epoch in range(1, epochs + 1):
            example_probs = None
            example_labels = None

            gen_loss = None
            disc_loss = None
            disc_accuracy = 0

            for i in tqdm(range(X.shape[0]), desc='epoch {} / {} : confidence {:.2f}'.format(str(epoch).zfill(4), str(epochs).zfill(4), confidence)):
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
                            
                            
                    # add a term that slightly skews real_Yi further to correct result
                    # epsilon = 0.01
                    # real_Yi = real_Yi * (1 - epsilon) + Y[i] * epsilon
                    
                    real_output = self.discriminator([X[i], real_Yi])
                    pred_output = self.discriminator([X[i], pred_Yi])
                    # pred_output2 = self.discriminator([X[i], tf.one_hot(tf.math.argmax(pred_Yi, axis=2), 4, axis=2)])
                    mismatch_output = self.discriminator([X[np.random.randint(0, len(X))], Y[i]])

                    gen_loss = generator_loss(pred_Yi, Y[i], pred_output)
                    disc_loss = discriminator_loss(real_output, tf.concat([pred_output, mismatch_output], axis=0))
                    # print(gen_loss.numpy(), disc_loss.numpy())
                    # print(pred_output.numpy(), real_output.numpy())
                    
                    disc_accuracy += (np.count_nonzero(pred_output < 0.5) + np.count_nonzero(real_output > 0.5)) / (X.shape[0] * X.shape[1] * 2)

                gradients_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                generator_optimizer.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))

                gradients_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
                discriminator_optimizer.apply_gradients(zip(gradients_discriminator, self.discriminator.trainable_variables))

            if not validation_data == (): gen_real_loss, gen_accuracy = self.generator.evaluate(validation_data[0], validation_data[1], verbose=0)
            else: gen_real_loss, gen_accuracy = self.generator.evaluate(X[0], Y[0], verbose=0)
            
            gen_losses.append(gen_loss.numpy())
            gen_real_losses.append(gen_real_loss)
            gen_accuracies.append(gen_accuracy)
            disc_losses.append(disc_loss.numpy())
            disc_accuracies.append(disc_accuracy)
            
            if epoch % print_interval == 0:
                # print(self.generator(X[0]))
                if validation_data == ():
                    gen_real_loss = str(gen_real_loss) + ' *'
                    gen_accuracy = str(gen_accuracy) + ' *'

                preprocessing.debug_print([
                    'epoch', f'{epoch:04}',
                    ':\n              generator GAN loss :', f'{gen_loss.numpy():05.5f}',
                    ' \n                  generator loss :', f'{gen_real_loss:05.5f}',
                    ' \n              generator accuracy :', f'{gen_accuracy:05.5f}',
                    ' \n          discriminator GAN loss :', f'{disc_loss.numpy():05.5f}',
                    ' \n          discriminator accuracy :', f'{disc_accuracy:05.5f}'
                ])
                
        if save:
            self.save_model()
    
        if plot:
            plt.plot(gen_losses, label='generator GAN loss')
            plt.plot(disc_losses, label='discriminator GAN loss')
            plt.plot(gen_real_losses, label='generator loss')
            plt.title('GAN loss')
            plt.ylabel('crossentropy loss')
            plt.xlabel('epoch')
            plt.legend()
            plt.show()
            
            plt.plot(gen_accuracies, label='generator accuracy')
            plt.plot(disc_accuracies, label='discriminator accuracy')
            plt.title('GAN accuracy')
            plt.ylabel('accuracy (0-1)')
            plt.xlabel('epoch')
            plt.legend()
            plt.show()
