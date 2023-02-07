# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/2/7 17:50
@brief: 构建模型的3种方法
"""
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import *
import matplotlib.pyplot as plt

train_token_path = './data/imdb/train_token.csv'
test_token_path = './data/imdb/test_token.csv'

MAX_WORDS = 10000
MAX_LEN = 200
BATCH_SIZE = 20


# 构建管道
def parse_line(line):
    t = tf.strings.split(line, '\t')
    label = tf.reshape(tf.cast(tf.strings.to_number(t[0]), tf.int32), (-1, ))
    features = tf.cast(tf.strings.to_number(tf.strings.split(t[1], ' ')), tf.int32)
    return features, label


ds_train = (tf.data.TextLineDataset(filenames=[train_token_path])
            .map(parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .shuffle(buffer_size=1000).batch(BATCH_SIZE)
            .prefetch(tf.data.experimental.AUTOTUNE))

ds_test = (tf.data.TextLineDataset(filenames=[test_token_path])
           .map(parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
           .shuffle(buffer_size=1000).batch(BATCH_SIZE)
           .prefetch(tf.data.experimental.AUTOTUNE))

# Sequential按层顺序构建模型
# tf.keras.backend.clear_session()
#
# model = models.Sequential()
# model.add(layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN))
# model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu'))
# model.add(layers.MaxPool1D(2))
# model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
# model.add(layers.MaxPool1D(2))
# model.add(layers.Flatten())
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
# model.summary()
#
# baselogger = callbacks.BaseLogger(stateful_metrics=['AUC'])
# logdir = './data/keras_model/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
# history = model.fit(ds_train, validation_data=ds_test, epochs=6, callbacks=[tensorboard_callback])


def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro--')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(['train_' + metric, 'val_' + metric])
    plt.show()
#
#
# plot_metric(history, 'auc')

# 函数式API创建任意结构模型
# tf.keras.backend.clear_session()
#
# inputs = layers.Input(shape=[MAX_LEN])
# x = layers.Embedding(MAX_WORDS, 7)(inputs)
#
# branch1 = layers.SeparableConv1D(64, 3, activation='relu')(x)
# branch1 = layers.MaxPool1D(3)(branch1)
# branch1 = layers.SeparableConv1D(32, 3, activation='relu')(branch1)
# branch1 = layers.GlobalMaxPool1D()(branch1)
#
# branch2 = layers.SeparableConv1D(64, 3, activation='relu')(x)
# branch2 = layers.MaxPool1D(3)(branch2)
# branch2 = layers.SeparableConv1D(32, 3, activation='relu')(branch2)
# branch2 = layers.GlobalMaxPool1D()(branch2)
#
# branch3 = layers.SeparableConv1D(64, 3, activation='relu')(x)
# branch3 = layers.MaxPool1D(3)(branch3)
# branch3 = layers.SeparableConv1D(32, 3, activation='relu')(branch3)
# branch3 = layers.GlobalMaxPool1D()(branch3)
#
# concat = layers.Concatenate()([branch1, branch2, branch3])
# outputs = layers.Dense(1, activation='sigmoid')(concat)
#
# model = models.Model(inputs=inputs, outputs=outputs)
# model.compile(optimizer='Nadam',
#               loss='binary_crossentropy',
#               metrics=['accuracy', 'AUC'])
#
# model.summary()
#
# logdir = './data/keras_model/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
# history = model.fit(ds_train, validation_data=ds_test, epochs=6, callbacks=[tensorboard_callback])
# plot_metric(history, 'auc')

# 自定义模型


# 先自定义一个残差模块，即自定义Layer
class ResBlock(layers.Layer):
    def __init__(self, kernel_size, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = layers.Conv1D(filters=64, kernel_size=self.kernel_size, activation='relu', padding='same')
        self.conv2 = layers.Conv1D(filters=32, kernel_size=self.kernel_size, activation='relu', padding='same')
        self.conv3 = layers.Conv1D(filters=input_shape[-1], kernel_size=self.kernel_size, activation='relu', padding='same')
        self.maxpool = layers.MaxPool1D(2)
        super(ResBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = layers.Add()([inputs, x])
        x = self.maxpool(x)
        return x

    def get_config(self):
        config = super(ResBlock, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


class ImdbModel(models.Model):
    def __init__(self):
        super(ImdbModel, self).__init__()

    def build(self, input_shape):
        self.embedding = layers.Embedding(MAX_WORDS, 7)
        self.block1 = ResBlock(7)
        self.block2 = ResBlock(5)
        self.dense = layers.Dense(1, activation='sigmoid')
        super(ImdbModel, self).build(input_shape)

    def call(self, x):
        x = self.embedding(x)
        x = self.block1(x)
        x = self.block2(x)
        x = layers.Flatten()(x)
        x = self.dense(x)
        return x


tf.keras.backend.clear_session()
model = ImdbModel()
model.build(input_shape=(None, 200))
model.summary()

model.compile(optimizer='Nadam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'AUC'])

logdir = './data/keras_model/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
history = model.fit(ds_train, validation_data=ds_test, epochs=6, callbacks=[tensorboard_callback])
plot_metric(history, 'auc')