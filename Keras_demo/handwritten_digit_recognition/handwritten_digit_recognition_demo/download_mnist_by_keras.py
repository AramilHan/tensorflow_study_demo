# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2022/12/23 20:03
@brief:
"""
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# 重复性设置
np.random.seed(1671)

# 网络和训练
NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10  # 输出个数等于数字个数
OPTIMIZER = SGD()  # SGD优化器
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2  # 训练集中用作验证集的数据比例

# 数据：混合并划分训练集和测试集数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train是60000行28 x 28的数据，变形为60000 x 784
RESHAPED = 784
print(len(x_train))
print(len(x_train[0]))
print(len(x_train[0][0]))
x_train = x_train.reshape(60000, RESHAPED)
x_test = x_test.reshape(10000, RESHAPED)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
# 归一化
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# 将类向量转换为二维类别矩阵
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)
# 使用softmax激活函数，聚合由10个神经元给出的10个答案
model = Sequential()
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED, )))
model.add(Activation('softmax'))
model.summary()
# 编译模型
# 目标函数选用categorical_crossentropy
# 优化器选用OPTIMIZER
# 性能评估函数选用accuracy
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
# 训练模型
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE,
                    nb_epoch=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
# 模型预测
score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print("\n")
print("Test score:", score[0])
print("Test accuracy:", score[1])
