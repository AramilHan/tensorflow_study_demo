# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/2/1 14:18
@brief: 低阶API示范
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 打印时间分割线
@tf.function
def printbar():
    today_ts = tf.timestamp() % (24 * 60 * 60)
    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minute = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format('{}', m)) == 1:
            return tf.strings.format('0{}', m)
        else:
            return tf.strings.format('{}', m)

    timestring = tf.strings.join([timeformat(hour), timeformat(minute), timeformat(second)], separator=':')
    tf.print('========'*8 + timestring)


# 样本数量
n = 400
# 生成测试用数据集
x = tf.random.uniform([n, 2], minval=-10, maxval=10)
w0 = tf.constant([[2.0], [-3.0]])
b0 = tf.constant([[3.0]])
# @表示矩阵乘法，增加正态扰动
y = x@w0 + b0 + tf.random.normal([n, 1], mean=0.0, stddev=2.0)

# 数据可视化
# plt.figure(figsize=(12, 5))
# ax1 = plt.subplot(121)
# ax1.scatter(x[:, 0], y[:, 0], c='b')
# plt.xlabel('x1')
# plt.ylabel('y', rotation=0)
#
# ax2 = plt.subplot(122)
# ax2.scatter(x[:, 1], y[:, 0], c='g')
# plt.xlabel('x2')
# plt.ylabel('y', rotation=0)
# plt.show()


# 构建数据管道迭代器
def data_iter(features, labels, batch_size=8):
    num_example = len(features)
    indices = list(range(num_example))
    # 样本的读取顺序是随机的
    np.random.shuffle(indices)
    for i in range(0, num_example, batch_size):
        indexes = indices[i: min(i + batch_size, num_example)]
        yield tf.gather(features, indexes), tf.gather(labels, indexes)


# 测试数据管道效果
# batch_size = 8
# (features, labels) = next(data_iter(x, y, batch_size))
# print(features)
# print(labels)


w = tf.Variable(tf.random.normal(w0.shape))
b = tf.Variable(tf.zeros_like(b0, dtype=tf.float32))


# 定义模型
class LinearRegression:
    # 正向传播
    def __call__(self, x):
        return x@w + b

    # 损失函数
    def loss_func(self, y_true, y_pred):
        return tf.reduce_mean((y_true - y_pred)**2/2)


model = LinearRegression()


# 使用动态图调试
@tf.function
def train_step(model, features, labels, learning_rate):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = model.loss_func(labels, predictions)
    # 反向传播求梯度
    dloss_dw, dloss_db = tape.gradient(loss, [w, b])
    w.assign(w - learning_rate*dloss_dw)
    b.assign(b - learning_rate*dloss_db)
    return loss


# 测试train_step效果
# batch_size = 10
# (features, labels) = next(data_iter(x, y, batch_size))
# test_loss = train_step(model, features, labels, learning_rate=0.001)
# print(test_loss)


def train_model(model, epochs):
    for epoch in tf.range(1, epochs+1):
        for features, labels in data_iter(x, y, 10):
            loss = train_step(model, features, labels, learning_rate=0.001)
        if epoch % 50 == 0:
            printbar()
            tf.print('epoch=', epoch, 'loss=', loss)
            tf.print('w=', w)
            tf.print('b=', b)


train_model(model, epochs=200)

# 结果可视化
plt.figure(figsize=(12, 5))
ax1 = plt.subplot(121)
ax1.scatter(x[:, 0], y[:, 0], c='b', label='samples')
ax1.plot(x[:, 0], w[0]*x[:, 0]+b[0], '-r', linewidth=5.0, label='model')
ax1.legend()
plt.xlabel('x1')
plt.ylabel('y', rotation=0)

ax2 = plt.subplot(122)
ax2.scatter(x[:, 1], y[:, 0], c='g', label='samples')
ax2.plot(x[:, 1], w[1]*x[:, 1]+b[0], '-r', linewidth=5.0, label='model')
ax2.legend()
plt.xlabel('x2')
plt.ylabel('y', rotation=0)
plt.show()
