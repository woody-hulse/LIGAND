import numpy as np
import tensorflow as tf

from utils import *

# Layers
class Transformer(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_size, ff_dim, dropout=0.1, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_size
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation='relu'),
            tf.keras.layers.Dense(input_shape[-1])
        ])
        self.dropout1 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=True):
        attn_output = self.attention(query=inputs, value=inputs, attention_mask=None, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
   
# Generators
class ActorMLP(tf.keras.Model):
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


class ActorVAE(tf.keras.Model):
    def __init__(self, input_shape, output_shape, latent_dim=32, num_transformers=3, hidden_size=32, name='actor_vae'):
        super().__init__(name=name)

        self.transformers = tf.keras.Sequential([Transformer(8, 8, hidden_size) for _ in range(num_transformers)])
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense_mean = tf.keras.layers.Dense(latent_dim)
        self.dense_log_var = tf.keras.layers.Dense(latent_dim)

        self.dense_decode1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense_decode2 = tf.keras.layers.Dense(output_shape[0] * output_shape[1], activation='relu')
        self.reshape = tf.keras.layers.Reshape(output_shape)
        self.dense_decode3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation='softmax'))

        self.sampling_layer = tf.keras.layers.Lambda(self.sampling, output_shape=(latent_dim,))

    def sampling(self, args):
        mean, log_var = args
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mean + tf.exp(0 * log_var) * epsilon

    def encode(self, x):
        x = self.transformers(x)
        x = self.flatten(x)
        x = self.dense1(x)
        mean = self.dense_mean(x)
        log_var = self.dense_log_var(x)
        return mean, log_var

    def decode(self, z):
        x = self.dense_decode1(z)
        x = self.dense_decode2(x)
        x = self.reshape(x)
        x = self.dense_decode3(x)
        return x

    def call(self, x):
        mean, log_var = self.encode(x)
        z = self.sampling_layer([mean, log_var])
        reconstructed = self.decode(z)
        return reconstructed

    def predict(self, x):
        return self.call(x)

class ActorTransformer1(tf.keras.Model):
    def __init__(self, input_shape, output_shape, num_transformers=3, hidden_size=32, name='actor_transformer_1'):
        super().__init__(name=name)

        self.transformers = tf.keras.Sequential([Transformer(8, 8, hidden_size) for _ in range(num_transformers)])
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_shape[0] * output_shape[1], activation='relu')
        self.reshape = tf.keras.layers.Reshape(output_shape)
        self.dense3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation='softmax'))

    def call(self, x):
        x = self.transformers(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)
        x = self.dense3(x)

        return x

    def predict(self, x):
        return self.call(x)

class ActorConvDeconv(tf.keras.Model):
    def __init__(self, input_shape, output_shape, latent_size=(20 * 4 - 6), name='actor_conv_deconv'):
        super().__init__(name=name)

        latent_size = output_shape[0] * 4 - 6

        self.Encoder = tf.keras.models.Sequential([
            tf.keras.layers.Reshape(list(input_shape) + [1,]),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_size)
        ])

        self.Decoder = tf.keras.models.Sequential([
            tf.keras.layers.Reshape((latent_size, 1)),
            tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, activation='relu', padding='valid'),
            tf.keras.layers.Conv1DTranspose(filters=8, kernel_size=3, activation='relu', padding='valid'),
            tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=3, activation='relu', padding='valid'),

            tf.keras.layers.Reshape(output_shape),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation='softmax', use_bias=False))
        ])
    
    def call(self, X):
        latent = self.Encoder(X)
        output = self.Decoder(latent)
        return output

    def predict(self, X):
        return self.call(X)
      
class ActorDense(tf.keras.Model):
    def __init__(self, input_shape, output_shape, reg=0.01, name='actor_dense'):
        super().__init__(name=name)

        self.Layers = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(output_shape[0] * 4, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Reshape(output_shape),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation='softmax'))
        ])
        
    def call(self, X):
        return self.Layers(X)

    def predict(self, X):
        return self.call(X)

# Discriminators
class CriticMLP(tf.keras.Model):
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
    
class CriticConv(tf.keras.Model):
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
    
class CriticTransformer1(tf.keras.Model):
    def __init__(self, input_shape, num_transformers=3, hidden_size=32, name='critic_transformer_1'):
        super().__init__(name=name)

        self.transformers = tf.keras.Sequential([Transformer(8, 4, hidden_size) for _ in range(num_transformers)])
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def preprocess_input(self, seqs, grna):
        output_shape = (seqs.shape[0], seqs.shape[1], grna.shape[2])
        pad_width = [(0, 0), (0, output_shape[1] - grna.shape[1]), (0, 0)]
        padded_grna = np.pad(grna, pad_width, mode='constant', constant_values=0)
        concat = np.concatenate([padded_grna, seqs], axis=2)
        x = concat[..., np.newaxis]
        return x

    def call(self, x):
        x = self.preprocess_input(*x)[:, :, :, 0]
        x = self.transformers(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        
        return x

    def predict(self, x):
        return self.call(x)
    
    
# Baselines
class GuessBaseline():
    def __init__(self, Y, name="guess_baseline"):
        self.shape = Y.shape[1:]
        self.Y = Y
        self.name = name

    def call(self, X):
        Y_pred = np.zeros((X.shape[0],) + self.shape)
        for i in range(X.shape[0]):
            Y_pred[i] = self.Y[np.random.randint(0, len(self.Y) - 1)]
        
        return Y_pred
    
    def predict(self, X):
        return self.call(X)
    
class CenterBaseline():
    def __init__(self, Y, name="center_baseline"):
        self.shape = Y.shape[1:]
        self.name = name

    def call(self, X):
        Y_pred = np.full((X.shape[0],) + self.shape, 1 / self.shape[1])
        return Y_pred
    
    def predict(self, X):
        return self.call(X)    
    
class MeanBaseline():
    def __init__(self, Y, name="mean_baseline"):
        self.pred = tf.reduce_mean(Y, axis=0)
        self.name = name

    def call(self, X):
        Y_pred = np.array([self.pred for _ in range(X.shape[0])])
        return Y_pred
    
    def predict(self, X):
        return self.call(X) 

class PairBaseline():
    def __init__(self, name="pair_baseline"):
        self.name = name

    def call(self, X):
        Y_pred = np.zeros((X.shape[0],) + (20, 4))
        for i in range(Y_pred.shape[0]):
            for j in range(Y_pred.shape[1]):
                if X[i][j][0] == 1: Y_pred[i][j][0] = 1
                if X[i][j][1] == 1: Y_pred[i][j][1] = 1
                if X[i][j][2] == 1: Y_pred[i][j][2] = 1
                if X[i][j][3] == 1: Y_pred[i][j][3] = 1
        return Y_pred
    
    def predict(self, X):
        return self.call(X)