# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2022/12/29 16:44
@brief: 深度神经网络识别CIFAR-10
"""
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt


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


# CIFAR-10是一个包含了60000张32 x 32像素的三通道图像数据集
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32
input_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
# 训练模型参数
BATCH_SIZE = 128
NB_EPOCH = 20
NB_EPOCH_DEEPER = 40
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIMIZER = RMSprop()
# 加载数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train sample')
print(x_test.shape[0], 'test sample')
# 转化成one-hot编码，并把图像归一化
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 使用基础深度学习网络训练模型
dlm = deep_learn_model()
# basic_model = dlm.build_basic_model(input_shape=input_shape, classes=NB_CLASSES)
basic_model = dlm.build_deeper_mode(input_shape=input_shape, classes=NB_CLASSES)
# 训练
basic_model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = basic_model.fit(x_train,
                          y_train,
                          batch_size=BATCH_SIZE,
                          epochs=NB_EPOCH_DEEPER,
                          validation_split=VALIDATION_SPLIT,
                          verbose=VERBOSE)
score = basic_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
print('Test score:', score[0])
print('Test accuracy:', score[1])
# 保存模型
model_json = basic_model.to_json()
open('cifar10_architecture.json', 'w').write(model_json)
# 保存模型参数
basic_model.save_weights('cifar10_weights.h5', overwrite=True)
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