import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from tqdm import tqdm

import preprocessing
from preprocessing import debug_print
from models import *
from utils import *


def discriminator_loss(real_output, pred_output, mismatch_output):
    lambda1 = 0.2
    lambda2 = 0.2
    lambda3 = 0.6
    # BC(1, D(g_t, d_t))
    real_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(real_output), real_output)
    # BC(0, D(G(d_t), d_t))
    fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(pred_output), pred_output)
    # BC(0, D(g_t, d_rand))
    mismatch_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(mismatch_output), mismatch_output)
    total_loss = real_loss * lambda1 + fake_loss * lambda2 + mismatch_loss * lambda3
    return total_loss

def generator_loss(pred, real, pred_output, mismatch_output):
    lambda1 = 0.4
    lambda2 = 0.5
    lambda3 = 0.1
    # BC(g_t, G(d_t))
    gen  = tf.keras.losses.CategoricalCrossentropy()(real, pred)
    # BC(1, D(G(d_t), d_t))
    disc = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(pred_output), pred_output)
    # BC(0, D(G(d_t), d_rand))
    disc_mismatch = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(pred_output), mismatch_output)
    return gen * lambda1 + disc * lambda2 + disc_mismatch * lambda3










class GAN(tf.keras.Model):
    def __init__(self, input_shape, output_shape, name='test_gan', generator = None, discriminator = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.shape = input_shape
        self.generator = generator
        self.discriminator = discriminator
        # self.generator = ActorMLP(output_shape)
        # self.generator = ActorTransformer1(input_shape, output_shape, num_transformers=8, hidden_size=64)
        self.generator.build((1,) + input_shape)
        # self.discriminator = CriticMLP()
        # self.discriminator = CriticTransformer1(input_shape, num_transformers=8, hidden_size=64)
        test_data = [np.zeros((1,) + input_shape), np.zeros((1,) + output_shape)]
        self.discriminator(test_data)
        
    
    def save_model(self, gen_losses, disc_losses, gen_real_losses, gen_accuracies, disc_accuracies):
        debug_print(['saving GAN'])
        self.generator.save_weights(f'models/{self.name}/generator.weights.h5')
        self.discriminator.save_weights(f'models/{self.name}/discriminator.weights.h5')

        df = pd.DataFrame({
            'gen_losses': gen_losses,
            'disc_losses': disc_losses,
            'gen_real_losses': gen_real_losses,
            'gen_accuracies': gen_accuracies,
            'disc_accuracies': disc_accuracies
        })
        df.to_csv(f'models/{self.name}/metrics.csv', index=False)    
    
    def load_model(self):
        debug_print(['loading GAN'])
        self.generator.load_weights(f'models/{self.name}/generator.weights.h5')
        self.discriminator.load_weights(f'models/{self.name}/discriminator.weights.h5')

    def plot(self, gen_losses, disc_losses, gen_real_losses, gen_accuracies, disc_accuracies):
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
        
    
    def generate(self, seqs):
        # introduce noise
        seqs += tf.random.normal(seqs.shape) * 0.1
        return self.generator(seqs)
    
    
    def get_real_Yi(self, pred_Yi, pred, real):
        real_Yi = pred_Yi.numpy()

        # define real_Yi (g_t) as the corrected probabilities from pred_Yi
        for m in range(real_Yi.shape[0]):
            for n in range(real_Yi.shape[1]):
                real_Yi[m][n][pred[m][n]], real_Yi[m][n][real[m][n]] = real_Yi[m][n][real[m][n]], real_Yi[m][n][pred[m][n]]
                
        return real_Yi
        
    
    def train(self, 
              X, Y, 
              epochs, 
              validation_data=(), 
              batch_size=8, 
              learning_rate=0.001, 
              print_interval=1, 
              summary=True, plot=True,
              save=True, load=False,
              name='test_gan'):
                
        if load:
            self.load_model()
            return
        
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
        gen_confidence = []
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
                # add noise to generator input
                noise = tf.random.normal(X[i].shape) * 0.01
                generated_probs = self.generator(X[i] + noise)

                # compute prediction confidence by deviation from mean prediction
                confidence = tf.keras.losses.MeanSquaredError()(generated_probs, mean_probs) * 100
                
                # generated sequences as the argmax of predicted input (unused)
                generated_sequences = tf.one_hot(tf.math.argmax(generated_probs, axis=2), 4, axis=2)

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    # get the generator output G(d_t)
                    pred_Yi = self.generator(X[i])
                    pred = np.argmax(pred_Yi, axis=2)
                    real = np.argmax(Y[i], axis=2)
                    real_Yi = self.get_real_Yi(pred_Yi, pred, real)
                    
                    # get the real (ground truth) and pred (generator) discriminator outputs
                    real_output = self.discriminator([X[i], real_Yi])
                    pred_output = self.discriminator([X[i], pred_Yi])
                    
                    # get the discriminator output for different permutations of seqs
                    mismatch_output_1 = self.discriminator([X[np.random.randint(0, len(X))], Y[i]])
                    indices = list(range(0, len(X[i])))
                    np.random.shuffle(indices)
                    mismatch_output_2 = self.discriminator([X[i][indices], Y[i]])
                    np.random.shuffle(indices)
                    mismatch_output_3 = self.discriminator([X[i][indices], pred_Yi])
                    mismatch_output = tf.concat([mismatch_output_1, mismatch_output_2, mismatch_output_3], axis=0)
                    gen_mismatch_output = self.discriminator([X[np.random.randint(0, len(X))], pred_Yi])

                    # compute loss:
                    # G_loss = lambda1 * BC(g_t, G(d_t)) + lambda2 * BC(1, D(G(d_t), d_t)) + lambda3 * BC(0, D(G(d_t), d_rand))
                    gen_loss = generator_loss(pred_Yi, Y[i], pred_output, gen_mismatch_output)
                    # D_loss = lambda1 * BC(1, D(g_t, d_t)) + lambda2 * (BC(0, D(G(d_t), d_t)) + BC(0, D(g_t, d_rand)))
                    disc_loss = discriminator_loss(real_output, pred_output, mismatch_output)
                    # print(gen_loss.numpy(), disc_loss.numpy())
                    # print(pred_output.numpy(), real_output.numpy())
                    
                    # compute accuracy of argmax of discriminator predicted output
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
            os.makedirs(f'models/{self.name}', exist_ok=True)
            self.save_model(gen_losses, disc_losses, gen_real_losses, gen_accuracies, disc_accuracies)
    
        if plot:
            self.plot(gen_losses, disc_losses, gen_real_losses, gen_accuracies, disc_accuracies)
            

class MLP_GAN(GAN):
    def __init__(self, input_shape, output_shape, name='mlp_gan', **kwargs):
        generator = ActorMLP(output_shape)
        discriminator = CriticMLP()

        super().__init__(input_shape, output_shape, name=name, 
                         generator=generator, discriminator=discriminator, **kwargs)
                         
class Conv_GAN(GAN):
    def __init__(self, input_shape, output_shape, name='conv_gan', **kwargs):
        generator = ActorConvDeconv(input_shape, output_shape)
        discriminator = CriticConv()

        super().__init__(input_shape, output_shape, name=name, 
                         generator=generator, discriminator=discriminator, **kwargs)
        
class Trans_Conv_GAN(GAN):
    def __init__(self, input_shape, output_shape, name='trans_conv_gan', **kwargs):
        generator = ActorTransformer1(input_shape, output_shape, num_transformers=8, hidden_size=64)
        discriminator = CriticConv()

        super().__init__(input_shape, output_shape, name=name, 
                         generator=generator, discriminator=discriminator, **kwargs)