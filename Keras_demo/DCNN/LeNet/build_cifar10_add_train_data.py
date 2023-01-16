# -*- encoding: utf-8 -*-
"""
@author:
@date: 2022/12/30 15:47
@brief:
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt
import numpy as np

NUM_TO_AUGMENT = 5


class deep_learn_model:
    def build_basic_model(self, input_shape, classes):
        """
        构建基础深度学习网络
        :param input_shape:
        :param classes:
        """
        model = Sequential()
        # Conv => Relu => Pool => Dropout
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.25))
        # 最后一层全连接网络
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        model.summary()
        return model

    def build_deeper_mode(self, input_shape, classes):
        """
        用深度学习网络改进性能
        :param input_shape:
        :param classes:
        :return:
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        model.summary()
        return model


# 加载数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 扩展数据集
print("扩展训练集")
datagen = ImageDataGenerator(
    rotation_range=40,  # 旋转图片的角度值(0 ~ 180)
    width_shift_range=0.2,  # 对图片做随机水平或垂直变化时的范围
    height_shift_range=0.2,
    zoom_range=0.2,  # 随机缩放图片的变化值
    horizontal_flip=True,  # 对选中的一半图片进行随机的水平翻转
    fill_mode='nearest',  # 图片翻转或交换后，用来填充新像素时采用的策略
)

xtas, ytas = [], []
for i in range(x_train.shape[0]):
    num_aug = 0
    x = x_train[i]
    x = x.reshape((1,) + x.shape)
    for x_aug in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='cifar', save_format='jpeg'):
        if num_aug >= NUM_TO_AUGMENT:
            break
        xtas.append(x_aug[0])
        num_aug += 1
# 匹配数据
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32
input_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
datagen.fit(x_train)
model = deep_learn_model().build_deeper_mode(input_shape=input_shape, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=50),
                              steps_per_epoch=x_train.shape[0],
                              epochs=50,
                              verbose=1)
score = model.evaluate(x_test, y_test, batch_size=50, verbose=1)
print("Test score:", score[0])
print("Test accuracy:", score[1])
