# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2022/12/23 17:13
@brief: 下载MNIST数据集
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images)
print(mnist.train.labels)
print(mnist.test.images)
print(mnist.test.labels)
# (x_test, y_test) = mnist.test()
# print(train)

