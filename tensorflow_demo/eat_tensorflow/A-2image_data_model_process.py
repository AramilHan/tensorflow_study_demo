# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/1/19 17:47
@brief: 图片数据建模流程范例
"""
import datetime
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from matplotlib import pyplot as plt
from pathlib import Path
from tensorboard import notebook
import pandas as pd

BATCH_SIZE = 100

"""
    数据读取
"""


def load_image(img_path, size=(32, 32)):
    label = tf.constant(1, tf.int8) if tf.strings.regex_full_match(img_path, '.*automobile.*') else tf.constant(0, tf.int8)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, size)/255.0
    return img, label


# 使用并行化处理num_parallel_calls和预存数据prefetch来提升性能
ds_train = (tf.data.Dataset.list_files('./data/cifar2/train/*/*.jpg')
            .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .shuffle(buffer_size=1000).batch(BATCH_SIZE)
            .prefetch(tf.data.experimental.AUTOTUNE))

ds_test = (tf.data.Dataset.list_files('./data/cifar2/test/*/*.jpg')
           .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
           .batch(BATCH_SIZE)
           .prefetch(tf.data.experimental.AUTOTUNE))

# 查看部分样本
# plt.figure(figsize=(8, 8))
# for i, (img, label) in enumerate(ds_train.unbatch().take(9)):
#     ax = plt.subplot(3, 3, i+1)
#     ax.imshow(img.numpy())
#     ax.set_title('label = %d' % label)
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.show()

for x, y in ds_train.take(1):
    print(x.shape, y.shape)

"""
    定义模型
"""
# 清空会话
tf.keras.backend.clear_session()

inputs = layers.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, kernel_size=(3, 3))(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(64, kernel_size=(5, 5))(x)
x = layers.MaxPool2D()(x)
x = layers.Dropout(rate=0.1)(x)
x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs=inputs, outputs=outputs)
model.summary()

"""
    训练模型
"""
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# logdir = os.path.join('data', 'autograph', stamp)
# python3建议使用pathlib修正
logdir = str(Path('./data/autograph', stamp))

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=["accuracy"])
history = model.fit(ds_train, epochs=10, validation_data=ds_test,
                    callbacks=[tensorboard_callback], workers=4)

"""
    评估模型
"""
# notebook.list()
# 在TensorBoard中查看模型
# notebook.start("--logdir {}".format(logdir))

df_history = pd.DataFrame(history.history)
df_history.index = range(1, len(df_history) + 1)
df_history.index.name = 'epoch'
print(df_history)


def plot_metric(history, metric):
    train_metric = history.history[metric]
    val_metric = history.history['val_' + metric]
    epochs = range(1, len(train_metric) + 1)
    plt.plot(epochs, train_metric, 'bo--')
    plt.plot(epochs, val_metric, 'ro--')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()


# plot_metric(history, "loss")
# plot_metric(history, "accuracy")

# 可以使用evaluate对数据进行评估
val_loss, val_accuracy = model.evaluate(ds_test, workers=4)
print(val_loss, val_accuracy)

"""
    使用模型
"""
pred = model.predict(ds_test)
print(pred)

for x, y in ds_test.take(1):
    print(model.predict_on_batch(x[0: 20]))

"""
    保存模型
"""
# 保存权重，该方式仅仅保存权重张量
model.save_weights('./data/tf_model_weights.ckpt', save_format="tf")
# 保存模型结构与模型参数到文件,该方式保存的模型具有跨平台性便于部署
model.save('./data/tf_model_savedmodel', save_format="tf")
print('export saved model.')

model_loaded = tf.keras.models.load_model('./data/tf_model_savedmodel')
model_loaded.evaluate(ds_test)
