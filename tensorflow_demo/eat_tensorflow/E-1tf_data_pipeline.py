# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/2/3 15:53
@brief: 数据管道
"""
import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
    numpy array 构建数据管道
"""
# iris = datasets.load_iris()
# ds1 = tf.data.Dataset.from_tensor_slices((iris['data'], iris['target']))
# for features, labels in ds1.take(5):
#     print(features, labels)

"""
    pandas DataFrame 构建数据管道
"""
# iris = datasets.load_iris()
# dfiris = pd.DataFrame(iris['data'], columns=iris.feature_names)
# ds2 = tf.data.Dataset.from_tensor_slices((dfiris.to_dict('list'), iris['target']))
# for features, label in ds2.take(3):
#     print(features, label)

"""
    python generator 构建数据管道
"""
# image_generator = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
#     "./data/cifar2/test/",
#     target_size=(32, 32),
#     batch_size=20,
#     class_mode='binary'
# )
# class_dict = image_generator.class_indices
# print(class_dict)
#
#
# def generator():
#     for features, label in image_generator:
#         yield (features, label)
#
#
# ds3 = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32))
# plt.figure(figsize=(6, 6))
# for i, (img, label) in enumerate(ds3.unbatch().take(9)):
#     ax = plt.subplot(3, 3, i+1)
#     ax.imshow(img.numpy())
#     ax.set_title('label=%d' % label)
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.show()

"""
    csv文件构建数据管道
"""
# ds4 = tf.data.experimental.make_csv_dataset(
#     file_pattern=['./data/titanic/train.csv', './data/titanic/test.csv'],
#     batch_size=3,
#     label_name='Survived',
#     na_value='',
#     num_epochs=1,
#     ignore_errors=True
# )
# for data, label in ds4.take(2):
#     print(data, label)

"""
    从文本文件构建数据管道
"""
# ds5 = tf.data.TextLineDataset(
#     filenames=['./data/titanic/train.csv', './data/titanic/test.csv']
# ).skip(1)
# for line in ds5.take(5):
#     print(line)

"""
    文件路径构建数据管道
"""
# ds6 = tf.data.Dataset.list_files('./data/cifar2/train/*/*.jpg')
# for file in ds6.take(5):
#     print(file)
#
#
# def load_image(img_path, size=(32, 32)):
#     label = 1 if tf.strings.regex_full_match(img_path, '.*/automobile/.*') else 0
#     img = tf.io.read_file(img_path)
#     img = tf.image.decode_jpeg(img)
#     img = tf.image.resize(img, size)
#     return img, label
#
#
# for i, (img, label) in enumerate(ds6.map(load_image).take(2)):
#     plt.figure(i)
#     plt.imshow((img / 255.0).numpy())
#     plt.title("label = %d" % label)
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

"""
    tfrecords文件构建数据管道
"""


# def create_tfrecords(inpath, outpath):
#     writer = tf.io.TFRecordWriter(outpath)
#     dirs = os.listdir(inpath)
#     for index, name in enumerate(dirs):
#         class_path = inpath + '/' + name + '/'
#         for img_name in os.listdir(class_path):
#             img_path = class_path + img_name
#             img = tf.io.read_file(img_path)
#             example = tf.train.Example(
#                 features=tf.train.Features(feature={
#                     'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#                     'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.numpy()]))
#                 })
#             )
#
#             writer.write(example.SerializeToString())
#     writer.close()
#
#
# create_tfrecords('./data/cifar2/test/', './data/cifar2_test.tfrecords')
#
#
# def parse_example(proto):
#     description = {'img_raw': tf.io.FixedLenFeature([], tf.string),
#                    'label': tf.io.FixedLenFeature([], tf.int64)}
#     example = tf.io.parse_single_example(proto, description)
#     img = tf.image.decode_jpeg(example['img_raw'])
#     img = tf.image.resize(img, (32, 32))
#     label = example['label']
#     return img, label
#
#
# ds7 = tf.data.TFRecordDataset('./data/cifar2_test.tfrecords').map(parse_example).shuffle(3000)
# for i, (img, label) in enumerate(ds7.take(9)):
#     ax = plt.subplot(3, 3, i + 1)
#     ax.imshow((img / 255.0).numpy())
#     ax.set_title("label = %d" % label)
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.show()

# ds = tf.data.Dataset.from_tensor_slices(['hello world', 'hello china', 'hello beijing'])
# ds_map = ds.map(lambda x: tf.strings.split(x, ' '))
# for x in ds_map:
#     print(x)

# ds = tf.data.Dataset.from_tensor_slices(['hello world', 'hello china', 'hello beijing'])
# ds_flatmap = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, ' ')))
# for x in ds_flatmap:
#     print(x)

# ds = tf.data.Dataset.from_tensor_slices(['hello world', 'hello china', 'hello beijing'])
# ds_interleave = ds.interleave(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, ' ')))
# for x in ds_interleave:
#     print(x)

# ds = tf.data.Dataset.from_tensor_slices(['hello world', 'hello china', 'hello beijing'])
# ds_filter = ds.filter(lambda x: tf.strings.regex_full_match(x, '.*[a|b].*'))
# for x in ds_filter:
#     print(x)

# ds1 = tf.data.Dataset.range(0, 3)
# ds2 = tf.data.Dataset.range(3, 6)
# ds3 = tf.data.Dataset.range(6, 9)
# ds_zip = tf.data.Dataset.zip((ds1, ds2, ds3))
# for x, y, z in ds_zip:
#     print(x, y, z)

# ds1 = tf.data.Dataset.range(0, 3)
# ds2 = tf.data.Dataset.range(3, 6)
# ds_concatenate = tf.data.Dataset.concatenate(ds1, ds2)
#
# for x in ds_concatenate:
#     print(x)

# ds = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
# result = ds.reduce(0, lambda x, y: tf.add(x, y))
# print(result)

# ds = tf.data.Dataset.range(12)
# ds_batch = ds.batch(4)
# for x in ds_batch:
#     print(x)

# elements = [[1, 2], [3, 4, 5], [6, 7], [8]]
# ds = tf.data.Dataset.from_generator(lambda: iter(elements), tf.int32)
# ds_padded_batch = ds.padded_batch(2, padded_shapes=[4, ])
# for x in ds_padded_batch:
#     print(x)

# ds = tf.data.Dataset.range(12)
# ds_window = ds.window(3, shift=1).flat_map(lambda x: x.batch(3, drop_remainder=True))
# for x in ds_window:
#     print(x)

# ds = tf.data.Dataset.range(12)
# ds_shuffle = ds.shuffle(buffer_size=5)
# for x in ds_shuffle:
#     print(x)

# ds = tf.data.Dataset.range(3)
# ds_repeat = ds.repeat(3)
# for x in ds_repeat:
#     print(x)

# ds = tf.data.Dataset.range(12)
# ds_shard = ds.shard(3, index=1)
# for x in ds_shard:
#     print(x)

# ds = tf.data.Dataset.range(12)
# ds_take = ds.take(3)
# tf.print(list(ds_take.as_numpy_iterator()))


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


# # 模拟数据准备
# def generator():
#     for i in range(10):
#         time.sleep(2)
#         yield i
#
#
# ds = tf.data.Dataset.from_generator(generator, output_types=tf.int32)
#
#
# # 模拟参数迭代
# def train_step():
#     time.sleep(1)
#
#
# printbar()
# tf.print(tf.constant("start training..."))
# for x in ds:
#     train_step()
# printbar()
# tf.print(tf.constant("end training..."))
#
# printbar()
# tf.print(tf.constant("start training with prefetch..."))
# for x in ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE):
#     train_step()
# printbar()
# tf.print(tf.constant("end training..."))

ds_files = tf.data.Dataset.list_files('./data/titanic/*.csv')
ds = ds_files.flat_map(lambda x: tf.data.TextLineDataset(x).skip(1))
for x in ds.take(4):
    print(x)

ds = ds_files.interleave(lambda x: tf.data.TextLineDataset(x).skip(1))
for x in ds.take(4):
    print(x)
