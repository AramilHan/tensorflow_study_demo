# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2022/12/27 16:11
@brief: 使用dropout改进简单网络
"""
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
np.random.seed(1671)  # 重复性设置

# 网络和训练
NB_EPOCH = 20  # 迭代次数
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10  # 输出个数等于数字个数
OPTIMIZER = Adam(lr=0.005)  # 优化器
N_HIDDEN = 128  # 隐藏层神经元个数
VALIDATION_SPLIT = 0.2  # 训练集中用作验证集的数据比例
DROPOUT = 0.3  # dropout丢弃概率

# 加载训练&测试数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
RESHAPED = 784
x_train = x_train.reshape(60000, RESHAPED)
x_test = x_test.reshape(10000, RESHAPED)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# 将类向量转换为二维类别矩阵
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# 定义输入层网络
model = Sequential()
# 定义输入层
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED, )))
# 定义第一层隐藏层激活函数
model.add(Activation('relu'))
# 定义第一层隐藏层后随机丢弃函数
model.add(Dropout(DROPOUT))
# 定义第二层隐藏层
model.add(Dense(N_HIDDEN))
# 定义第二层隐藏层激活函数
model.add(Activation('relu'))
# 定义第二层隐藏层后随机丢弃函数
model.add(Dropout(DROPOUT))
# 定义输出层
model.add(Dense(NB_CLASSES))
# 定义输出层激活函数
model.add(Activation('softmax'))
model.summary()

# 模型编译
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])
# 模型训练
history = model.fit(x_train,
                    y_train,
                    nb_epoch=NB_EPOCH,
                    batch_size=BATCH_SIZE,
                    verbose=VERBOSE,
                    validation_split=VALIDATION_SPLIT)
# 模型测试集评估
score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print("\n")
print("Test score:", score[0])
print("Test accuracy:", score[1])
predict_class = model.predict_classes(x_test, verbose=VERBOSE)
print(predict_class)
