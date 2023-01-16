# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2022/12/28 17:10
@brief: 模型检查点设置
"""
from __future__ import division, print_function

import keras.callbacks
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
import numpy as np
import os

BATCH_SIZE = 128
NUM_EPOCHS = 20
MODEL_DIR = "/tmp"

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(512, input_shape=(784,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 保存最好的模型
checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR, "model-{epoch:02d}.h5"))
print(os.path.join(MODEL_DIR, "model-{epoch:02d}.h5"))
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.1, callbacks=[checkpoint])
keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=False)
