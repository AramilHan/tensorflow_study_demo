# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/2/1 15:11
@brief: 低阶API——DNN
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf


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


# 正负样本数量
n_positive, n_negative = 2000, 2000
# 生成正样本，小圆环分布
r_p = 5.0 + tf.random.truncated_normal([n_positive, 1], 0.0, 1.0)
theta_p = tf.random.uniform([n_positive, 1], 0.0, 2 * np.pi)
xp = tf.concat([r_p * tf.cos(theta_p), r_p * tf.sin(theta_p)], axis=1)
yp = tf.ones_like(r_p)

# 生成负样本，大圆环分布
r_n = 8.0 + tf.random.truncated_normal([n_negative, 1], 0.0, 1.0)
theta_n = tf.random.uniform([n_negative, 1], 0.0, 2 * np.pi)
xn = tf.concat([r_n * tf.cos(theta_n), r_n * tf.sin(theta_n)], axis=1)
yn = tf.zeros_like(r_n)

# 汇总样本
x = tf.concat([xp, xn], axis=0)
y = tf.concat([yp, yn], axis=0)


# 可视化
# plt.figure(figsize=(6, 6))
# plt.scatter(xp[:, 0].numpy(), xp[:, 1].numpy(), c='r')
# plt.scatter(xn[:, 0].numpy(), xn[:, 1].numpy(), c='g')
# plt.legend(['positive', 'negative'])
# plt.show()


# 构建数据管道迭代器
def date_iter(features, labels, batch_size=8):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 样本的读取顺序是随机的
    np.random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        indexes = indices[i: min(i + batch_size, num_examples)]
        yield tf.gather(features, indexes), tf.gather(labels, indexes)


# 测试管道效果
# batch_size = 10
# (features, labels) = next(date_iter(x, y, batch_size))
# print(features)
# print(labels)


class DNNModel(tf.Module):
    def __init__(self, name=None):
        super(DNNModel, self).__init__(name=name)
        self.w1 = tf.Variable(tf.random.truncated_normal([2, 4]), dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([1, 4]), dtype=tf.float32)
        self.w2 = tf.Variable(tf.random.truncated_normal([4, 8]), dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([1, 8]), dtype=tf.float32)
        self.w3 = tf.Variable(tf.random.truncated_normal([8, 1]), dtype=tf.float32)
        self.b3 = tf.Variable(tf.zeros([1, 1]), dtype=tf.float32)

    # 正向传播
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)])
    def __call__(self, x):
        x = tf.nn.relu(x @ self.w1 + self.b1)
        x = tf.nn.relu(x @ self.w2 + self.b2)
        y = tf.nn.sigmoid(x @ self.w3 + self.b3)
        return y

    # 损失函数（二元交叉熵）
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, 1], dtype=tf.float32)])
    def loss_func(self, y_true, y_pred):
        # 将预测值限制在1e-7以上，1-1e-7以下，避免log(0)错误
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        bce = - y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(bce)

    # 评估指标（准确率）

    def metric_func(self, y_true, y_pred):
        y_pred = tf.where(y_pred > 0.5, tf.ones_like(y_pred, dtype=tf.float32),
                          tf.zeros_like(y_pred, dtype=tf.float32))
        acc = tf.reduce_mean(1 - tf.abs(y_true - y_pred))
        return acc


model = DNNModel()


# 测试模型结构
# batch_size = 10
# (features, labels) = next(date_iter(x, y, batch_size))
# predictions = model(features)
# loss = model.loss_func(labels, predictions)
# metric = model.metric_func(labels, predictions)
# tf.print('init loss:', loss)
# tf.print('init metric:', metric)
# print(len(model.trainable_variables))


# 使用Autograph机制转换成静态图加速
@tf.function
def train_step(model, features, labels, learning_rate):
    # 正向传播求损失
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = model.loss_func(labels, predictions)
    # 反向传播求梯度
    grads = tape.gradient(loss, model.trainable_variables)
    # 执行梯度下降
    for p, dloss_dp in zip(model.trainable_variables, grads):
        p.assign(p - learning_rate * dloss_dp)
    # 计算评估指标
    metric = model.metric_func(labels, predictions)
    return loss, metric


def train_model(model, epochs):
    for epoch in tf.range(1, epochs + 1):
        for features, labels in date_iter(x, y, 100):
            loss, metric = train_step(model, features, labels, learning_rate=0.001)
        if epoch % 100 == 0:
            printbar()
            tf.print("epoch=", epoch, 'loss=', loss, 'metric=', metric)


train_model(model, epochs=600)

# 结果可视化
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax1.scatter(xp[:, 0], xp[:, 1], c="r")
ax1.scatter(xn[:, 0], xn[:, 1], c="g")
ax1.legend(["positive", "negative"])
ax1.set_title("y_true")

xp_pred = tf.boolean_mask(x, tf.squeeze(model(x) >= 0.5), axis=0)
xn_pred = tf.boolean_mask(x, tf.squeeze(model(x) < 0.5), axis=0)

ax2.scatter(xp_pred[:, 0], xp_pred[:, 1], c="r")
ax2.scatter(xn_pred[:, 0], xn_pred[:, 1], c="g")
ax2.legend(["positive", "negative"])
ax2.set_title("y_pred")
plt.show()
