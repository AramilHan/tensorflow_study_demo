# -*- encoding: utf-8 -*-
"""
@author: Aramil
@date: 2022/8/26 15:56
@brief: 使用Keras训练感知机
"""
from keras.models import Sequential
from keras.layers import Dense

perceptron_model = Sequential()
perceptron_model.add(Dense(12, input_dim=8, kernel_initializer='random_uniform'))
