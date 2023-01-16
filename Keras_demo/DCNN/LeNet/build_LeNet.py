# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2022/12/29 15:48
@brief: 构建LeNet
"""
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt


class LeNet:
    def build(self, input_shape, classes):
        model = Sequential()
        # Conv => ReLu => Pool
        model.add(Conv2D(20, kernel_size=5, padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        # Conv => ReLu => Pool
        model.add(Conv2D(50, kernel_size=5, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        # 最后一层使用标准的全连接网络
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model


NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
IMG_ROWS, IMG_COLS = 28, 28
NB_CLASSES = 10
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)
# 获取训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
K.set_image_data_format("channels_last")
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
# 需要使用形状60000 x 【1 x 28 x 28】作为卷积网络的输入
x_train = x_train[:, np.newaxis, :, :]
x_test = x_test[:, np.newaxis, :, :]
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)
# 初始化优化器和模型
lt = LeNet()
leNet_model = lt.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
leNet_model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = leNet_model.fit(x_train,
                          y_train,
                          batch_size=BATCH_SIZE,
                          epochs=NB_EPOCH,
                          verbose=VERBOSE,
                          validation_split=VALIDATION_SPLIT)
score = leNet_model.evaluate(x_test, y_test, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])
# 列出全部历史数据
print(history.history.keys())
# 汇总准确率历史数据
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# 汇总损失函数历史数据
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
