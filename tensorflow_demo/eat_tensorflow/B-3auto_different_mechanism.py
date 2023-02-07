# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/1/31 17:58
@brief: 自动微分机制
"""
import tensorflow as tf
import numpy as np

"""
    利用梯度磁带求导数
"""
# f(x) = a*x**2 + b*x + c
# x = tf.Variable(0.0, name='x', dtype=tf.float32)
# a = tf.constant(1.0)
# b = tf.constant(-2.0)
# c = tf.constant(1.0)
# with tf.GradientTape() as tape:
#     y = a * tf.pow(x, 2) + b * x + c
# dy_dx = tape.gradient(y, x)
# print(dy_dx)
"""
    对常量求导
"""
# with tf.GradientTape() as tape:
#     tape.watch([a, b, c])
#     y = a * tf.pow(x, 2) + b * x + c
# dy_dx, dy_da, dy_db, dy_dc = tape.gradient(y, [x, a, b, c])
# print(dy_dx)
# print(dy_da)
# print(dy_db)
# print(dy_dc)
"""
    对二阶导数求导
"""
# with tf.GradientTape() as tape2:
#     with tf.GradientTape() as tape1:
#         y = a * tf.pow(x, 2) + b * x + c
#     dy_dx = tape1.gradient(y, x)
# dy2_dx2 = tape2.gradient(dy_dx, x)
# print(dy2_dx2)
"""
    在autograph使用
"""


# @tf.function
# def f(x):
#     a = tf.constant(1.0)
#     b = tf.constant(-2.0)
#     c = tf.constant(1.0)
#     x = tf.cast(x, tf.float32)
#     with tf.GradientTape() as tape:
#         tape.watch(x)
#         y = a * tf.pow(x, 2) + b * x + c
#     dy_dx = tape.gradient(y, x)
#     return (dy_dx, y)
#
#
# tf.print(f(tf.constant(0.0)))
# tf.print(f(tf.constant(0.2)))

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
# for _ in range(1000):
#     with tf.GradientTape() as tape:
#         y = a * tf.pow(x, 2) + b * x + c
#     dy_dx = tape.gradient(y, x)
#     optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])
# tf.print("y=", y, "; x=", x)

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
#
#
# def f():
#     y = a * tf.pow(x, 2) + b * x + c
#     return y
#
#
# for _ in range(1000):
#     optimizer.minimize(f, [x])
# tf.print("y=", f(), "; x=", x)

# x = tf.Variable(0.0, name='x', dtype=tf.float32)
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
#
#
# @tf.function
# def minimizef():
#     a = tf.constant(1.0)
#     b = tf.constant(-2.0)
#     c = tf.constant(1.0)
#
#     for _ in range(1000):
#         with tf.GradientTape() as tape:
#             y = a * tf.pow(x, 2) + b * x + c
#         dy_dx = tape.gradient(y, x)
#         optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])
#     y = a * tf.pow(x, 2) + b * x + c
#     return y
#
#
# tf.print(minimizef())
# tf.print(x)

x = tf.Variable(0.0, name='x', dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


@tf.function
def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c
    return y


@tf.function
def train(epoch):
    for _ in tf.range(epoch):
        optimizer.minimize(f, [x])
    return f()


tf.print(train(1000))
tf.print(x)
