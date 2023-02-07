# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/2/2 15:20
@brief: 高阶API示范——DNN二分类模型
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


# 正负样本数量
n_positive, n_negative = 2000, 2000
n = n_positive + n_negative
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

ds_train = (tf.data.Dataset.from_tensor_slices((x[0: n*3 // 4, :], y[0: n*3 // 4, :]))
            .shuffle(buffer_size=1000)
            .batch(20)
            .prefetch(tf.data.experimental.AUTOTUNE)
            .cache())

ds_valid = (tf.data.Dataset.from_tensor_slices((x[n*3 // 4:, :], y[n*3 // 4:, :]))
            .batch(20)
            .prefetch(tf.data.experimental.AUTOTUNE)
            .cache())


tf.keras.backend.clear_session()


class DNNModel(models.Model):
    def __init__(self):
        super(DNNModel, self).__init__()

    def build(self, input_shape):
        self.dense1 = layers.Dense(4, activation='relu', name='dense1')
        self.dense2 = layers.Dense(8, activation='relu', name='dense2')
        self.dense3 = layers.Dense(1, activation='sigmoid', name='dense3')
        super(DNNModel, self).build(input_shape)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)])
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        y = self.dense3(x)
        return y


model = DNNModel()
model.build(input_shape=(None, 2))
model.summary()

optimizer = optimizers.Adam(learning_rate=0.01)
loss_func = tf.keras.losses.BinaryCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_metric = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_metric = tf.keras.metrics.BinaryAccuracy(name='valid_accuracy')


@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = loss_func(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)


def valid_step(model, features, labels):
    predictions = model(features)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)


def train_model(model, ds_train, ds_valid, epochs):
    for epoch in tf.range(1, epochs+1):
        for features, labels in ds_train:
            train_step(model, features, labels)
        for features, labels in ds_valid:
            valid_step(model, features, labels)

        logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'
        if epoch % 100 == 0:
            printbar()
            tf.print(tf.strings.format(logs,
                                       (epoch, train_loss.result(), train_metric.result(), valid_loss.result(), valid_metric.result())))

        train_loss.reset_states()
        train_metric.reset_states()
        valid_loss.reset_states()
        valid_metric.reset_states()


train_model(model, ds_train, ds_valid, 1000)

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
