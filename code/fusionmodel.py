from glob import glob
import random
from PIL import Image
import numpy as np
from tensorflow.python.keras.layers.merge import multiply
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
import pickle


class CrossTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(CrossTransformerBlock, self).__init__()
        self.att1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.att2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.att_fusion = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim*2, activation=tf.nn.gelu), layers.Dense(embed_dim*2),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'att': self.att,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
        })
        return config

    def call(self, inputs1, inputs2, training):
        m1 = self.att1(inputs1, inputs2)
        m1 = self.layernorm1(inputs1 + m1)

        m2 = self.att2(inputs2, inputs1)
        m2 = self.layernorm1(inputs2 + m2)

        attn_output = tf.concat([m1, m2], axis=2)
        out = self.att_fusion(attn_output, attn_output)
#         out = self.dropout1(out, training=training)
        out = self.layernorm2(out + attn_output)
        ffn_output = self.ffn(out)
#         ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm3(out + ffn_output)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(
                ff_dim, activation=tf.nn.gelu
                ), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'att': self.att,
            'ffn': self.ffn,
            'att_scores': self.att_scores,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
        })
        return config

    def call(self, inputs, training):
        # attn_output = self.att(inputs, inputs)
        attn_output, attn_scores = self.att(inputs, inputs, return_attention_scores=True)
#       attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), attn_scores

class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.positions = tf.range(start=0, limit=maxlen, delta=1)
    def call(self, x):
        positions = self.pos_emb(self.positions)
        return x + positions
    
class FeatureEmbedding(layers.Layer):
    #input shape: (batch_size, seq_len, embed_dim)
    #output shape: (batch_size, seq_len, embed_dim)
    #seq_len = 85

    def __init__(self, num_hid):
        super().__init__()
        self.zero_pad1 = tf.keras.layers.ZeroPadding1D(padding=42)
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 2, strides=1, padding="same", activation='relu'
        )
        self.zero_pad2 = tf.keras.layers.ZeroPadding1D(padding=42)
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 3, strides=1, padding="same", activation='relu'
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 3, strides=1, padding="same", activation='relu'
        )

    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'conv1': self.conv1,
            'conv2': self.conv2,
            'conv3': self.conv3,
        })
        return config

    def call(self, x):
        # x = self.zero_pad1(x)
        x = self.conv1(x)
        # x = self.zero_pad2(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        return x


# class FeatureEmbedding(layers.Layer):
#     def __init__(self, num_hid):
#         super(FeatureEmbedding, self).__init__()
#         self.ffn = tf.keras.layers.Dense(num_hid, activation=tf.nn.relu)
#         # self.ffn2 = tf.keras.layers.Dense(num_hid, activation=tf.nn.relu)

#     def call(self, x):
#         x = self.ffn(x)
#         return x


def lr_step_decay(epoch, lr):
    import math

    initial_learning_rate = 2e-3
    drop_rate = 0.1
    epochs_drop = 15.0
    
    return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))

def get_embeddings(x, embed_dim):
    input_shape = x.shape
    embedding_layer = FeatureEmbedding(embed_dim)
    embedding = embedding_layer(x)
    print(embedding.shape)
    position_embedding = PositionEmbedding(embedding.shape[1], embedding.shape[2])
    x = position_embedding(embedding)
    return x
















def transformer_fusion(MAX_SEQ_LEN, lr=2e-4,loss='binary_crossentropy', dropout=0.5):

    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 88))
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 272))

    x1_embed = get_embeddings(x1_inputs, 8)
    x2_embed = get_embeddings(x2_inputs, 8)

    x1 = TransformerBlock(8, 1, 8)(x1_embed)
    x2 = TransformerBlock(8, 1, 8)(x2_embed)

    x3 = CrossTransformerBlock(8, 1, 8)(x1, x2)
    x3 = layers.GlobalAveragePooling1D()(x3)
    x3 = layers.Dropout(dropout)(x3)
    x3 = layers.Dense(2)(x3)

    y_depression_pred = layers.Softmax(name='depression')(x3)
    model = keras.Model(inputs=[x1_inputs, x2_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])

    return model

def transformer_fusion_daic(MAX_SEQ_LEN, lr=2e-4):

    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 25))
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 136))

    x1_embed = get_embeddings(x1_inputs, 4)
    x2_embed = get_embeddings(x2_inputs, 4)

    x1 = TransformerBlock(4, 1, 4)(x1_embed)
    x2 = TransformerBlock(4, 1, 4)(x2_embed)

    x3 = CrossTransformerBlock(4, 1, 4)(x1, x2)
    x3 = layers.GlobalAveragePooling1D()(x3)
    x3 = layers.Dropout(0.75)(x3)
    x3 = layers.Dense(2)(x3)

    y_depression_pred = layers.Softmax(name='depression')(x3)
    model = keras.Model(inputs=[x1_inputs, x2_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])

    return model




def transformer_concat(MAX_SEQ_LEN, lr=2e-4):

    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 88)) #audio
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 272)) #video
    x3_inputs = layers.Input(shape=(MAX_SEQ_LEN, 768)) #text

    x1_embed = get_embeddings(x1_inputs, 8)
    x2_embed = get_embeddings(x2_inputs, 8)
    x3_embed = get_embeddings(x3_inputs, 8)

    x1 = TransformerBlock(8, 1, 8)(x1_embed)
    x2 = TransformerBlock(8, 1, 8)(x2_embed)
    x3 = TransformerBlock(8, 1, 8)(x3_embed)

    x3 = layers.concatenate([x1, x2, x3], axis=1)
    x3 = layers.GlobalAveragePooling1D()(x3)
    x3 = layers.Dropout(0.5)(x3)
    x3 = layers.Dense(2)(x3)

    y_depression_pred = layers.Softmax(name='depression')(x3)
    model = keras.Model(inputs=[x1_inputs, x2_inputs, x3_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])

    return model

def cross_transformer_concat(MAX_SEQ_LEN, lr=2e-4):

    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 88)) #audio
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 272)) #video
    x3_inputs = layers.Input(shape=(MAX_SEQ_LEN, 768)) #text
    
    x1_embed = get_embeddings(x1_inputs, 4)
    x2_embed = get_embeddings(x2_inputs, 4)
    x3_embed = get_embeddings(x3_inputs, 4)

    x1 = CrossTransformerBlock(4, 1, 4)(x1_embed, x2_embed)
    x1 = layers.GlobalAveragePooling1D()(x1)
    x2 = CrossTransformerBlock(4, 1, 4)(x2_embed, x3_embed)
    x2 = layers.GlobalAveragePooling1D()(x2)
    x3 = CrossTransformerBlock(4, 1, 4)(x3_embed, x1_embed)
    x3 = layers.GlobalAveragePooling1D()(x3)
    
    x3 = layers.concatenate([x1, x2, x3],axis=1)
    x3 = layers.Dropout(0.5)(x3)
    x3 = layers.Dense(2)(x3)


    y_depression_pred = layers.Softmax(name='depression')(x3)
    model = keras.Model(inputs=[x1_inputs, x2_inputs, x3_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])

    return model


def transformer_add(MAX_SEQ_LEN, lr=2e-4):

    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 25))
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 136))

    x1_embed = get_embeddings(x1_inputs, 8)
    x2_embed = get_embeddings(x2_inputs, 8)

    x1 = TransformerBlock(8, 1, 8)(x1_embed)
    x2 = TransformerBlock(8, 1, 8)(x2_embed)

    x3 = layers.Add()([x1, x2])
    x3 = layers.GlobalAveragePooling1D()(x3)
    x3 = layers.Dropout(0.5)(x3)
    x3 = layers.Dense(2)(x3)

    y_depression_pred = layers.Softmax(name='depression')(x3)
    model = keras.Model(inputs=[x1_inputs, x2_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])

    return model


def transformer_avg(MAX_SEQ_LEN, lr=2e-4):

    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 25))
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 136))

    x1_embed = get_embeddings(x1_inputs, 8)
    x2_embed = get_embeddings(x2_inputs, 8)

    x1 = TransformerBlock(8, 1, 8)(x1_embed)
    x2 = TransformerBlock(8, 1, 8)(x2_embed)

    x3 = layers.Average()([x1, x2])
    x3 = layers.GlobalAveragePooling1D()(x3)
    x3 = layers.Dropout(0.5)(x3)
    x3 = layers.Dense(2)(x3)

    y_depression_pred = layers.Softmax(name='depression')(x3)
    model = keras.Model(inputs=[x1_inputs, x2_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])

    return model

def transformer_multiply(MAX_SEQ_LEN, lr=2e-4):

    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 25))
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 136))

    x1_embed = get_embeddings(x1_inputs, 8)
    x2_embed = get_embeddings(x2_inputs, 8)

    x1 = TransformerBlock(8, 1, 8)(x1_embed)
    x2 = TransformerBlock(8, 1, 8)(x2_embed)

    x3 = layers.Multiply()([x1, x2])
    x3 = layers.GlobalAveragePooling1D()(x3)
    x3 = layers.Dropout(0.5)(x3)
    x3 = layers.Dense(2)(x3)

    y_depression_pred = layers.Softmax(name='depression')(x3)
    model = keras.Model(inputs=[x1_inputs, x2_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])

    return model

def audio_model(MAX_SEQ_LEN, lr=2e-4,loss='binary_crossentropy', dropout=0.5):
    
    x_inputs = layers.Input(shape=(MAX_SEQ_LEN, 88))
    x = get_embeddings(x_inputs, 8)
    x = TransformerBlock(8, 1, 8)(x)
    x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Dropout(dropout)(x)
    y_deprssion_pred = layers.Dense(2, activation="softmax", name='depression')(x)

    model = keras.Model(inputs=[x_inputs], outputs=[y_deprssion_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])

    return model

def visual_model(MAX_SEQ_LEN, lr=2e-4,loss='binary_crossentropy', dropout=0.5):
    
    x_inputs = layers.Input(shape=(MAX_SEQ_LEN, 272))
    x = get_embeddings(x_inputs, 8)
    x = TransformerBlock(8, 1, 8)(x)
    x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Dropout(dropout)(x)
    y_deprssion_pred = layers.Dense(2, activation="softmax", name='depression')(x)

    model = keras.Model(inputs=[x_inputs], outputs=[y_deprssion_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])

    return model

def text_model(MAX_SEQ_LEN, lr=2e-4,loss='binary_crossentropy', dropout=0.5):
    
    x_inputs = layers.Input(shape=(MAX_SEQ_LEN, 768))
    x = get_embeddings(x_inputs, 8)
    x = TransformerBlock(8, 1, 8)(x)
    x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Dropout(dropout)(x)
    y_deprssion_pred = layers.Dense(2, activation="softmax", name='depression')(x)

    model = keras.Model(inputs=[x_inputs], outputs=[y_deprssion_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])

    return model

# Model baselines.

def bidirectional_lstm_fusion(MAX_SEQ_LEN, lr=2e-4):

    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 25))
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 136))

    x1 = layers.Bidirectional(layers.LSTM(8, return_sequences=False))(x1_inputs)
    x2 = layers.Bidirectional(layers.LSTM(8, return_sequences=False))(x2_inputs)
    
    x = layers.concatenate([x1, x2])
    x = layers.Dropout(0.5)(x)
    y_depression_pred = layers.Dense(2, activation="softmax", name='depression')(x)
    # y_depression_pred = layers.Softmax(name='depression')(x_multiply)

    model = keras.Model(inputs=[x1_inputs, x2_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])
    
    return model

def conv1d_fusion(MAX_SEQ_LEN, lr=2e-4):

    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 25))
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 136))

    x1 = layers.Conv1D(8, 2, 2)(x1_inputs)
    x1 = layers.Conv1D(8, 3, 2)(x1)
    x2 = layers.Conv1D(8, 2, 2)(x2_inputs)
    x2 = layers.Conv1D(8, 3, 2)(x2)
    
    x = layers.concatenate([x1, x2])
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    y_depression_pred = layers.Dense(2, activation="softmax", name='depression')(x)
    # y_depression_pred = layers.Softmax(name='depression')(x_multiply)

    model = keras.Model(inputs=[x1_inputs, x2_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])
    
    return model


def tensor_fusion(MAX_SEQ_LEN, lr=2e-4):

    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 25))
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 136))

    x1_embed = layers.Dense(8, activation='relu')(x1_inputs)
#     x1_embed = layers.Dense(16, activation='relu')(x1_embed)
    x1 = tf.math.reduce_mean(x1_embed, axis=1)
    
    x2_embed = layers.Dense(8, activation='relu')(x2_inputs)
    # x2_embed = layers.Dense(8, activation='relu')(x2_embed)
    x2 = tf.math.reduce_mean(x2_embed, axis=1)

    m1 = layers.concatenate([x1, tf.ones((tf.shape(x1)[0], 1))])
    m1 = tf.reshape(m1, shape=(-1, 9, 1))
    m2 = layers.concatenate([x2, tf.ones((tf.shape(x2)[0], 1))])
    m2 = tf.reshape(m2, shape=(-1, 1, 9))

    x_multiply = tf.reshape(layers.Multiply()([m1, m2]), shape=(-1, 9, 9, 1))

    x_flat = layers.Flatten()(x_multiply)
    x_flat = layers.Dense(16, activation='relu')(x_flat)
#     x_flat = layers.Dense(32, activation='relu')(x_flat)
    x_flat = layers.Dense(16, activation='relu')(x_flat)
    x_flat = layers.Dropout(0.5)(x_flat)
    
    y_depression_pred = layers.Dense(2, activation="softmax", name='depression')(x_flat)
    # y_depression_pred = layers.Softmax(name='depression')(x_multiply)

    model = keras.Model(inputs=[x1_inputs, x2_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])

    return model





####### Additional Model #######

def transformer_fusion_avt(MAX_SEQ_LEN, lr=2e-4,loss='binary_crossentropy', dropout=0.5):
    
    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 88))
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 272))
    x3_inputs = layers.Input(shape=(MAX_SEQ_LEN, 768))
    

    x1_embed = get_embeddings(x1_inputs, 4)
    x2_embed = get_embeddings(x2_inputs, 4)
    x3_embed = get_embeddings(x3_inputs, 4)

    x1 = TransformerBlock(4, 1, 4)(x1_embed)
    x1 = TransformerBlock(4, 1, 4)(x1)
    x2 = TransformerBlock(4, 1, 4)(x2_embed)
    x2 = TransformerBlock(4, 1, 4)(x2)

    #attention score
    # x1_prob = layers.Dense(4, activation='softmax')(x1)
    # x1 = layers.Multiply()([x1, x1_prob])

    # x2_prob = layers.Dense(4, activation='softmax')(x2)
    # x2= layers.Multiply()([x2, x2_prob])

    # x3_prob = layers.Dense(4, activation='softmax')(x3_embed)
    # x3_embed = layers.Multiply()([x3_embed, x3_prob]) 


    x_av = CrossTransformerBlock(4, 1, 4)(x1, x2)
    x_av = layers.GlobalAveragePooling1D()(x_av)

    x_at = CrossTransformerBlock(4, 1, 4)(x1, x3_embed)
    x_at = layers.GlobalAveragePooling1D()(x_at)

    x_vt = CrossTransformerBlock(4, 1, 4)(x2, x3_embed)
    x_vt = layers.GlobalAveragePooling1D()(x_vt)

    x3 = tf.concat([x_av, x_at,x_vt], axis=1)

    x3 = layers.Dropout(dropout)(x3)
    
    y_depression_pred = layers.Dense(2, activation="softmax", name='depression')(x3)
    # y_depression_pred = layers.Dense(1, activation='sigmoid')(x3)

    model = keras.Model(inputs=[x1_inputs, x2_inputs, x3_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['accuracy'])

    return model

def multitask_transformer_fusion(MAX_SEQ_LEN, lr=2e-4):
    
    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 25))
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 136))

    x1_embed = get_embeddings(x1_inputs, 4)
    x2_embed = get_embeddings(x2_inputs, 4)

    x1 = TransformerBlock(4, 1, 4)(x1_embed)
    x2 = TransformerBlock(4, 1, 4)(x2_embed)

    x3 = CrossTransformerBlock(4, 1, 4)(x1, x2)
    x3 = layers.GlobalAveragePooling1D()(x3)

    y_depression_pred = layers.Dense(1, name='depression')(x3)
    y_gender_pred = layers.Dense(2, activation='softmax', name='gender')(x3)

    # y_depression_pred = layers.Sigmoid(name='depression')(x3)
    model = keras.Model(inputs=[x1_inputs, x2_inputs], outputs=[y_depression_pred, y_gender_pred])

    model.compile(optimizer=RMSprop(lr=lr),
                loss={
                    'depression': tf.keras.losses.MeanAbsoluteError(), 
                    'gender': tf.keras.losses.CategoricalCrossentropy()
                    },
                loss_weights={
                    'depression': 0.3,
                    'gender': 0.7
                },
                metrics={
                    'depression': tf.keras.metrics.RootMeanSquaredError(),
                    'gender': tf.keras.metrics.Accuracy()
                })

    return model



def hique(MAX_SEQ_LEN, lr=2e-4,loss='binary_crossentropy', dropout=0.5):
    
    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 88)) #audio
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 272)) #video
    x3_inputs = layers.Input(shape=(MAX_SEQ_LEN, 768)) #text
    

    x1_embed = get_embeddings(x1_inputs, 4)
    x2_embed = get_embeddings(x2_inputs, 4)
    x3_embed = get_embeddings(x3_inputs, 4)

    x1 = TransformerBlock(4, 1, 4)(x1_embed)
    x1 = TransformerBlock(4, 1, 4)(x1)
    x2 = TransformerBlock(4, 1, 4)(x2_embed)
    x2 = TransformerBlock(4, 1, 4)(x2)
    x3 = TransformerBlock(4, 1, 4)(x3_embed)
    x3 = TransformerBlock(4, 1, 4)(x3)


    x_av = CrossTransformerBlock(4, 1, 4)(x1, x2)
    x_av = layers.GlobalAveragePooling1D()(x_av)

    x_at = CrossTransformerBlock(4, 1, 4)(x1, x3)
    x_at = layers.GlobalAveragePooling1D()(x_at)

    x_vt = CrossTransformerBlock(4, 1, 4)(x2, x3)
    x_vt = layers.GlobalAveragePooling1D()(x_vt)


    x4 = tf.concat([x_av, x_at, x_vt], axis=1)

    x4 = layers.Dropout(dropout)(x4)
    y_depression_pred = layers.Dense(2,activation='softmax', name='depression')(x4)

    model = keras.Model(inputs=[x1_inputs, x2_inputs, x3_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['accuracy'])

    return model


def transformer_fusion_new(MAX_SEQ_LEN, lr=2e-4,loss='binary_crossentropy', dropout=0.5):
    
    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 88)) #audio
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 272)) #video
    x3_inputs = layers.Input(shape=(MAX_SEQ_LEN, 768)) #text
    

    x1_embed = get_embeddings(x1_inputs, 4)
    x2_embed = get_embeddings(x2_inputs, 4)
    x3_embed = get_embeddings(x3_inputs, 4)

    x1,att_score_audio_1 = TransformerBlock(4, 1, 4)(x1_embed)
    x1,att_score_audio_2 = TransformerBlock(4, 1, 4)(x1)
    x2,att_score_video_1 = TransformerBlock(4, 1, 4)(x2_embed)
    x2,att_score_video_2 = TransformerBlock(4, 1, 4)(x2)
    x3,att_score_text_1 = TransformerBlock(4, 1, 4)(x3_embed)
    x3,att_score_text_2 = TransformerBlock(4, 1, 4)(x3)

    # x1 = TransformerBlock(4, 1, 4)(x1_embed)
    # x1 = TransformerBlock(4, 1, 4)(x1)
    # x2 = TransformerBlock(4, 1, 4)(x2_embed)
    # x2 = TransformerBlock(4, 1, 4)(x2)
    # x3 = TransformerBlock(4, 1, 4)(x3_embed)
    # x3 = TransformerBlock(4, 1, 4)(x3)

    x_concat = tf.concat([x1, x2, x3], axis=1)
    x_concat, att_score_concat_1 = TransformerBlock(4, 1, 4)(x_concat)
    x_concat, att_score_concat_2 = TransformerBlock(4, 1, 4)(x_concat)
    # x_concat = TransformerBlock(4, 1, 4)(x_concat)
    # x_concat = TransformerBlock(4, 1, 4)(x_concat)

    x_av = CrossTransformerBlock(4, 1, 4)(x1, x2)
    x_av = layers.GlobalAveragePooling1D()(x_av)

    x_at = CrossTransformerBlock(4, 1, 4)(x1, x3)
    x_at = layers.GlobalAveragePooling1D()(x_at)

    x_vt = CrossTransformerBlock(4, 1, 4)(x2, x3)
    x_vt = layers.GlobalAveragePooling1D()(x_vt)

    x_concat = layers.GlobalAveragePooling1D()(x_concat)

    # x4 = tf.concat([x_av, x_at, x_vt], axis=1)
    x4 = tf.concat([x_concat, x_av, x_at, x_vt], axis=1)

    x4 = layers.Dropout(dropout)(x4)
    y_depression_pred = layers.Dense(2,activation='softmax', name='depression')(x4)

    model = keras.Model(inputs=[x1_inputs, x2_inputs, x3_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['accuracy'])

    return model


def audio_model_lld(MAX_SEQ_LEN, lr=2e-4,loss='categorical_crossentropy', dropout=0.5):
    
    x_inputs = layers.Input(shape=(MAX_SEQ_LEN, 25))
    x = get_embeddings(x_inputs, 8)
    x = TransformerBlock(8, 1, 8)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    y_deprssion_pred = layers.Dense(2, activation="softmax", name='depression')(x)

    model = keras.Model(inputs=[x_inputs], outputs=[y_deprssion_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])

    return model


def audio_bidirectional_lstm(MAX_SEQ_LEN, lr=2e-4,loss='categorical_crossentropy', dropout=0.5):
    
    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 88))

    x1 = layers.Bidirectional(layers.LSTM(8, return_sequences=False))(x1_inputs)
    
    x = layers.Dropout(0.5)(x1)
    y_depression_pred = layers.Dense(2, activation="softmax", name='depression')(x)

    model = keras.Model(inputs=[x1_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])
    
    return model






def transformer_fusion_ori(MAX_SEQ_LEN, lr=2e-4,loss='binary_crossentropy', dropout=0.5):
    
    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 25))
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 136))
    x3_inputs = layers.Input(shape=(768))
    

    x1_embed = get_embeddings(x1_inputs, 4)
    x2_embed = get_embeddings(x2_inputs, 4)

    x1 = TransformerBlock(4, 1, 4)(x1_embed)
    x1 = TransformerBlock(4, 1, 4)(x1)
    x2 = TransformerBlock(4, 1, 4)(x2_embed)
    x2 = TransformerBlock(4, 1, 4)(x2)

    x3 = CrossTransformerBlock(4, 1, 4)(x1, x2)
    x3 = layers.GlobalAveragePooling1D()(x3)

    x3 = tf.concat([x3, x3_inputs], axis=1)

    x3 = layers.Dense(64, activation='relu')(x3)
    # x3 = layers.Dropout(0.5)(x3)
    # x3 = layers.Dense(128, activation='relu')(x3)

    y_depression_pred = layers.Dense(2,activation='softmax', name='depression')(x3)

    # y_depression_pred = layers.Sigmoid(name='depression')(x3)
    model = keras.Model(inputs=[x1_inputs, x2_inputs, x3_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])

    return model



def transformer_fusion_vt(MAX_SEQ_LEN, lr=2e-4,loss='binary_crossentropy', dropout=0.5):
    
    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 272))
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 768))

    x1_embed = get_embeddings(x1_inputs, 8)
    x2_embed = get_embeddings(x2_inputs, 8)

    x1 = TransformerBlock(8, 1, 8)(x1_embed)
    # x2 = TransformerBlock(8, 1, 8)(x2_embed)

    x3 = CrossTransformerBlock(8, 1, 8)(x1, x2_embed)

    
    x3 = layers.GlobalAveragePooling1D()(x3)
    # x3 = layers.Dropout(dropout)(x3)
    x3 = layers.Dense(2)(x3)

    y_depression_pred = layers.Softmax(name='depression')(x3)
    model = keras.Model(inputs=[x1_inputs, x2_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])

    return model


def transformer_fusion_at(MAX_SEQ_LEN, lr=2e-4,loss='binary_crossentropy', dropout=0.5):
    
    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 88))
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 768))

    x1_embed = get_embeddings(x1_inputs, 8)
    x2_embed = get_embeddings(x2_inputs, 8)

    x1 = TransformerBlock(8, 1, 8)(x1_embed)
    x2 = TransformerBlock(8, 1, 8)(x2_embed)

    x3 = CrossTransformerBlock(8, 1, 8)(x1, x2)
    x3 = layers.GlobalAveragePooling1D()(x3)
    # x3 = layers.Dropout(dropout)(x3)
    x3 = layers.Dense(2)(x3)

    y_depression_pred = layers.Softmax(name='depression')(x3)
    model = keras.Model(inputs=[x1_inputs, x2_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['acc'])

    return model




def transformer_fusion_avt_lld(MAX_SEQ_LEN, lr=2e-4,loss='binary_crossentropy', dropout=0.5):
    
    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 25))
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 272))
    x3_inputs = layers.Input(shape=(MAX_SEQ_LEN, 768))
    

    x1_embed = get_embeddings(x1_inputs, 4)
    x2_embed = get_embeddings(x2_inputs, 4)
    x3_embed = get_embeddings(x3_inputs, 4)

    x1 = TransformerBlock(4, 1, 4)(x1_embed)
    x1 = TransformerBlock(4, 1, 4)(x1)
    x2 = TransformerBlock(4, 1, 4)(x2_embed)
    x2 = TransformerBlock(4, 1, 4)(x2)

    x_av = CrossTransformerBlock(4, 1, 4)(x1, x2)
    x_av = layers.GlobalAveragePooling1D()(x_av)

    x_at = CrossTransformerBlock(4, 1, 4)(x1, x3_embed)
    x_at = layers.GlobalAveragePooling1D()(x_at)

    x_ta = CrossTransformerBlock(4, 1, 4)(x3_embed, x1)
    x_ta = layers.GlobalAveragePooling1D()(x_ta)

    x_va = CrossTransformerBlock(4, 1, 4)(x2, x1)
    x_va = layers.GlobalAveragePooling1D()(x_va)

    x_vt = CrossTransformerBlock(4, 1, 4)(x2, x3_embed)
    x_vt = layers.GlobalAveragePooling1D()(x_vt)

    x_tv = CrossTransformerBlock(4, 1, 4)(x3_embed, x2)
    x_tv = layers.GlobalAveragePooling1D()(x_tv)

    x3 = tf.concat([x_av, x_at,x_vt,x_ta,x_va,x_tv], axis=1)

    x3 = layers.Dropout(dropout)(x3)
    
    y_depression_pred = layers.Dense(2, activation="softmax", name='depression')(x3)
    # y_depression_pred = layers.Dense(1, activation='sigmoid')(x3)

    model = keras.Model(inputs=[x1_inputs, x2_inputs, x3_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['accuracy'])
    

    return model





def transformer_fusion_avt_l(MAX_SEQ_LEN, lr=2e-4,loss='binary_crossentropy', dropout=0.5):

    x1_inputs = layers.Input(shape=(MAX_SEQ_LEN, 88))
    x2_inputs = layers.Input(shape=(MAX_SEQ_LEN, 272))
    x3_inputs = layers.Input(shape=(MAX_SEQ_LEN, 768))
    

    x1_embed = get_embeddings(x1_inputs, 4)
    x2_embed = get_embeddings(x2_inputs, 4)
    x3_embed = get_embeddings(x3_inputs, 4)

    x1 = TransformerBlock(4, 1, 4)(x1_embed)
    x1 = TransformerBlock(4, 1, 4)(x1)
    x2 = TransformerBlock(4, 1, 4)(x2_embed)
    x2 = TransformerBlock(4, 1, 4)(x2)
    x3 = TransformerBlock(4, 1, 4)(x3_embed)
    x3 = TransformerBlock(4, 1, 4)(x3)

    # x_av = CrossTransformerBlock(4, 1, 4)(x1, x2)
    # x_av = layers.GlobalAveragePooling1D()(x_av)

    # x_at = CrossTransformerBlock(4, 1, 4)(x1, x3)
    # x_at = layers.GlobalAveragePooling1D()(x_at)

    # x_vt = CrossTransformerBlock(4, 1, 4)(x2, x3)
    # x_vt = layers.GlobalAveragePooling1D()(x_vt)

    x_tv = CrossTransformerBlock(4, 1, 4)(x3, x2)
    x_tv = layers.GlobalAveragePooling1D()(x_tv)

    # x3 = tf.concat([x_tv, x_vt], axis=1)

    # x3 = layers.Dropout(dropout)(x3)
    
    y_depression_pred = layers.Dense(2, activation="softmax", name='depression')(x_tv)
    # y_depression_pred = layers.Dense(1, activation='sigmoid')(x3)

    model = keras.Model(inputs=[x1_inputs, x2_inputs, x3_inputs], outputs=[y_depression_pred])

    model.compile(optimizer=Adam(lr=lr),
                loss="categorical_crossentropy", metrics=['accuracy'])
    

    return model

