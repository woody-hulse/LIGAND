import csv
import os
from tqdm import tqdm
import datetime

import math
import random
import numpy as np
import pandas as pd

import tensorflow as tf


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
    

class ActorTransformer1(tf.keras.Model):
    def __init__(self, input_shape, output_shape, num_transformers=3, hidden_size=32, name='actor_transformer_1'):
        super().__init__()

        embedding_dim = 4
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=4, output_dim=embedding_dim, input_length=input_shape[0])
        self.transformers = tf.keras.Sequential([Transformer(8, 8, hidden_size) for _ in range(num_transformers)])
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_shape[0] * output_shape[1], activation='relu')
        self.reshape = tf.keras.layers.Reshape(output_shape)
        self.dense3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation='softmax'))

    def call(self, x):
        x = self.embedding_layer(x)
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
            tf.keras.layers.Reshape(input_shape + (1,)),
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
                if X[i][j][0] == 1: Y_pred[i][j][3] = 1
                if X[i][j][1] == 1: Y_pred[i][j][2] = 1
                if X[i][j][2] == 1: Y_pred[i][j][1] = 1
                if X[i][j][3] == 1: Y_pred[i][j][0] = 1
        return Y_pred
    
    def predict(self, X):
        return self.call(X)