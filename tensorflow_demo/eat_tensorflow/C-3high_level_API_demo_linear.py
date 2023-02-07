# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/2/2 11:16
@brief: 高阶API示范——线性回归模型
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, losses, metrics, optimizers


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

# 定义模型
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(1, input_shape=(2,)))
model.summary()

# 训练模型
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x, y, batch_size=10, epochs=200)
tf.print('w=', model.layers[0].kernel)
tf.print('b=', model.layers[0].bias)

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
