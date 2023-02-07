# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/1/31 10:31
@brief: 张量数据结构
"""
import numpy as np
import tensorflow as tf

# # tf.int32 类型常量
# i = tf.constant(1)
# # tf.int64 类型常量
# l = tf.constant(1, dtype=tf.int64)
# # tf.float32 类型常量
# f = tf.constant(1.23)
# # tf.double 类型常量
# d = tf.constant(3.14, dtype=tf.double)
# # tf.string 类型常量
# s = tf.constant('hello world')
# # tf.bool 类型常量
# b = tf.constant(True)
#
# print(tf.int32 == np.int32)
# print(tf.int64 == np.int64)
# print(tf.float32 == np.float)
# print(tf.double == np.float64)
# print(tf.string == np.unicode)
# print(tf.bool == np.bool)

"""
    张量维度
"""
# # 标量，0维张量
# scalar = tf.constant(True)
# print(tf.rank(scalar))
# print(scalar.numpy().ndim)
# # 向量，1维张量
# vector = tf.constant([1, 2, 3, 4])
# print(tf.rank(vector))
# print(np.ndim(vector.numpy()))
# # 矩阵，2维张量
# matrix = tf.constant([[1, 2], [3, 4]])
# print(tf.rank(matrix))
# print(np.ndim(matrix.numpy()))
# # 图片，3维张量
# tensor3 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# print(tensor3)
# print(tf.rank(tensor3))
# print(np.ndim(tensor3.numpy()))
# # 视频，4维张量
# tensor4 = tf.constant([[[[1, 1], [2, 2]], [[3, 3], [4, 4]]], [[[5, 5], [6, 6]], [[7, 7], [8, 8]]]])
# print(tensor4)
# print(tf.rank(tensor4))
# print(np.ndim(tensor4.numpy()))
# # 使用tf.cast改变张量的数据类型
# h = tf.constant([123, 456])
# f = tf.cast(h, tf.float32)
# print(h.dtype, f.dtype)
# tf -> np
# y = tf.constant([[1, 2], [3, 4]])
# print(y.numpy())
# print(y.shape)
# u = tf.constant(u'你好 约翰')
# print(u.numpy())
# print(u.numpy().decode('utf-8'))

# c = tf.constant([1, 2])
# print(c)
# print(id(c))
# c = c + tf.constant([1, 2])
# print(c)
# print(id(c))

v = tf.Variable([1, 2], name='v')
print(v)
print(id(v))
v.assign_add([1, 2])
print(v)
print(id(v))
