# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/1/30 16:37
@brief: 时间序列数据建模
"""
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, losses, metrics, callbacks
from pathlib import Path

"""
    准备数据
"""
# 读取数据
df = pd.read_csv('./data/covid_19/covid-19.csv', sep='\t')
# df.plot(x='date', y=['confirmed_num', 'cured_num', 'dead_num'], figsize=(10, 6))
# plt.xticks(rotation=60)
# plt.show()

dfdata = df.set_index("date")
dfdiff = dfdata.diff(periods=1).dropna()
dfdiff = dfdiff.reset_index("date")
#
# dfdiff.plot(x="date", y=["confirmed_num", "cured_num", "dead_num"], figsize=(10, 6))
# plt.xticks(rotation=60)
dfdiff = dfdiff.drop("date", axis=1).astype("float32")
# plt.show()

WINDOW_SIZE = 8


def batch_dataset(dataset):
    dataset_batched = dataset.batch(WINDOW_SIZE, drop_remainder=True)
    return dataset_batched


ds_data = tf.data.Dataset.from_tensor_slices(tf.constant(dfdiff.values, dtype=tf.float32)).window(WINDOW_SIZE,
                                                                                                  shift=1).flat_map(
    batch_dataset)
ds_label = tf.data.Dataset.from_tensor_slices(tf.constant(dfdiff.values[WINDOW_SIZE:], dtype=tf.float32))
# 数据较小，可以将全部训练数据放入到一个batch中，提升性能
ds_train = tf.data.Dataset.zip((ds_data, ds_label)).batch(38).cache()

"""
    定义模型
"""


# 考虑到新增确诊，新增治愈，新增死亡人数数据不可能小于0，设计如下结构
class Block(layers.Layer):
    def __init__(self, **kwargs):
        super(Block, self).__init__(**kwargs)

    def call(self, x_input, x):
        x_out = tf.maximum((1 + x) * x_input[:, -1, :], 0.0)
        return x_out

    def get_config(self):
        config = super(Block, self).get_config()
        return config


tf.keras.backend.clear_session()
x_input = layers.Input(shape=(None, 3), dtype=tf.float32)
x = layers.LSTM(3, return_sequences=True, input_shape=(None, 3))(x_input)
x = layers.LSTM(3, return_sequences=True, input_shape=(None, 3))(x)
x = layers.LSTM(3, return_sequences=True, input_shape=(None, 3))(x)
x = layers.LSTM(3, input_shape=(None, 3))(x)
x = layers.Dense(3)(x)

x = Block()(x_input, x)
model = models.Model(inputs=[x_input], outputs=[x])
model.summary()

"""
    训练模型
"""


# 自定义损失函数，考虑平方差和预测目标的比值
class MSPE(losses.Loss):
    def call(self, y_true, y_prod):
        err_percent = (y_true - y_prod) ** 2 / (tf.maximum(y_true ** 2, 1e-7))
        mean_err_percent = tf.reduce_mean(err_percent)
        return mean_err_percent

    def get_config(self):
        config = super(MSPE, self).get_config()
        return config


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=MSPE(name='MSPE'))
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = str(Path('./data/autograph/' + stamp))

tb_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
# 如果loss在100个epoch后没有提升，学习率减半。
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=100)
# 当loss在200个epoch后没有提升，则提前终止训练。
stop_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=200)
callbacks_list = [tb_callback, lr_callback, stop_callback]

history = model.fit(ds_train, epochs=500, callbacks=callbacks_list)

"""
    评估模型
"""

# def plot_metric(history, metric):
#     train_metrics = history.history[metric]
#     epochs = range(1, len(train_metrics) + 1)
#     plt.plot(epochs, train_metrics, 'bo--')
#     plt.title('Training '+ metric)
#     plt.xlabel("Epochs")
#     plt.ylabel(metric)
#     plt.legend(["train_"+metric])
#     plt.show()
#
#
# plot_metric(history, "loss")
"""
    使用模型
"""
# 使用dfresult记录现有数据以及此后预测的疫情数据
dfresult = dfdiff[["confirmed_num", "cured_num", "dead_num"]].copy()
dfresult.tail()
# 预测此后100天的新增走势,将其结果添加到dfresult中
for i in range(100):
    arr_predict = model.predict(tf.constant(tf.expand_dims(dfresult.values[-38:, :], axis=0)))

    dfpredict = pd.DataFrame(tf.cast(tf.floor(arr_predict), tf.float32).numpy(),
                             columns=dfresult.columns)
    dfresult = dfresult.append(dfpredict, ignore_index=True)

dfresult.query("confirmed_num==0").head()
# 第55天开始新增确诊降为0，第45天对应3月10日，也就是10天后，即预计3月20日新增确诊降为0
# 注：该预测偏乐观
dfresult.query("cured_num==0").head()
# 第164天开始新增治愈降为0，第45天对应3月10日，也就是大概4个月后，即7月10日左右全部治愈。
# 注: 该预测偏悲观，并且存在问题，如果将每天新增治愈人数加起来，将超过累计确诊人数。
dfresult.query("dead_num==0").head()
# 第60天开始，新增死亡降为0，第45天对应3月10日，也就是大概15天后，即20200325
# 该预测较为合理

model.save('./data/tf_model_savedmodel', save_format="tf")
print('export saved model.')
model_loaded = tf.keras.models.load_model('./data/tf_model_savedmodel', compile=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model_loaded.compile(optimizer=optimizer, loss=MSPE(name="MSPE"))
model_loaded.predict(ds_train)
