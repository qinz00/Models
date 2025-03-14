import gc
import os
import random

import h5py
import numpy as np
from keras.layers import Dense, Conv1D, ELU,Flatten, Dropout, BatchNormalization, MaxPooling1D, Add
from keras.models import Model
from keras.layers import Input

from keras.callbacks import ModelCheckpoint
from keras.optimizer_v2.adam import Adam
# from keras.optimizers import Adam
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import tensorflow as tf
from tensorflow import keras


def get_positional_embedding(sentence_length, d_model):
    angle_rads = get_angles(np.arange(sentence_length)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    position_embedding = np.zeros((sentence_length, d_model))
    position_embedding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    position_embedding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    position_embedding = position_embedding[np.newaxis, ...]
    return tf.cast(position_embedding, dtype = tf.float32)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, ((2 * i) / np.float32(d_model)))
    return pos * angle_rates


def scaled_dot_product_attention(q, k, v, mask):

    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logit = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logit += (mask * -1e9)


    attention_weights = tf.nn.softmax(scaled_attention_logit, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights




class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = self.d_model // self.num_heads

        self.WQ = keras.layers.Dense(self.d_model)
        self.WK = keras.layers.Dense(self.d_model)
        self.WV = keras.layers.Dense(self.d_model)

        self.dense = keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):


        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.WQ(q)
        k = self.WK(k)
        v = self.WV(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention_outputs, attention_weights = \
            scaled_dot_product_attention(q, k, v, mask)

        scaled_attention_outputs = tf.transpose(
            scaled_attention_outputs, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention_outputs, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights



# Feedforwared
def feed_forward_network(d_model, dff):
    return keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),
        keras.layers.Dense(d_model)
    ])


def pre_net(d_model):
    return keras.layers.Dense(d_model)


# Encode Layers
class EncoderLayer(keras.layers.Layer):


    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, dff)

        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        
        self.add1 = keras.layers.Add()
        self.add2 = keras.layers.Add()

    def call(self, x, training, encoder_padding_mask):
        
        ln_x = self.layer_norm1(x)
        attn_output, _ = self.mha(ln_x, ln_x, ln_x, encoder_padding_mask)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.add1([x, attn_output])
        
        ln_x2 = self.layer_norm2(out1)
        ffn_output = self.ffn(ln_x2)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.add2([out1, ffn_output])

        return out2





## EncoderModel
class EncoderModel_6(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, sentence_length, rate=0.1, **kwargs):
        super(EncoderModel_6, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.sentence_length = sentence_length

        self.position_embedding = get_positional_embedding(self.sentence_length, self.d_model)

        self.dropout = keras.layers.Dropout(rate)
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(self.num_layers)]

    def call(self, x, training, encoder_padding_mask):
        input_seq_len = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x += self.position_embedding[:, :input_seq_len, :]


        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, encoder_padding_mask)

        return x


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate,
            'sentence_length':self.sentence_length
        })
        return config



