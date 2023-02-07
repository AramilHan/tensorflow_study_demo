# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/2/1 17:14
@brief: 中阶API示范——线性回归
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, losses, metrics, optimizers


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
    tf.print('========' * 8 + timestring)


# 样本数量
n = 400
# 生成测试用数据集
x = tf.random.uniform([n, 2], minval=-10, maxval=10)
w0 = tf.constant([[2.0], [-3.0]])
b0 = tf.constant([[3.0]])
y = x @ w0 + b0 + tf.random.normal([n, 1], mean=0.0, stddev=2.0)

# 数据可视化
# plt.figure(figsize=(12, 5))
# ax1 = plt.subplot(121)
# ax1.scatter(x[:, 0], y[:, 0], c="b")
# plt.xlabel("x1")
# plt.ylabel("y", rotation=0)
#
# ax2 = plt.subplot(122)
# ax2.scatter(x[:, 1], y[:, 0], c="g")
# plt.xlabel("x2")
# plt.ylabel("y", rotation=0)
# plt.show()

# 构建输入数据管道
ds = (tf.data.Dataset.from_tensor_slices((x, y))
      .shuffle(buffer_size=100).batch(10)
      .prefetch(tf.data.experimental.AUTOTUNE))

# 定义模型
model = layers.Dense(units=1)
# 用build方法创建Variable
model.build(input_shape=(2,))
model.loss_func = losses.mean_squared_error
model.optimizer = optimizers.SGD(learning_rate=0.001)


# 使用Autograph机制转换静态图加速
@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = model.loss_func(tf.reshape(labels, [-1]), tf.reshape(predictions, [-1]))
    grads = tf.gradients(loss, model.variables)
    model.optimizer.apply_gradients(zip(grads, model.variables))
    return loss


# 测试train_step效果
# features, labels = next(ds.as_numpy_iterator())
# train_step(model, features, labels)


def train_model(model, epochs):
    for epoch in tf.range(1, epochs+1):
        loss = tf.constant(0.0)
        for features, labels in ds:
            loss = train_step(model, features, labels)
        if epoch % 50 == 0:
            printbar()
            tf.print('epoch=', epoch, 'loss=', loss)
            tf.print('w=', model.variables[0])
            tf.print('b=', model.variables[1])


train_model(model, epochs=200)

# 结果可视化
w, b = model.variables
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
