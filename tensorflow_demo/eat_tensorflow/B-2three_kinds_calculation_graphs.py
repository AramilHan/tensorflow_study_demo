# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/1/31 16:23
@brief: 三种计算图
"""
import tensorflow as tf
"""
    TensorFlow2.0使用tf1.0方式构建静态计算图
"""
# g = tf.compat.v1.Graph()
# # 定义计算图
# with g.as_default():
#     x = tf.compat.v1.placeholder(name='x', shape=[], dtype=tf.string)
#     y = tf.compat.v1.placeholder(name='y', shape=[], dtype=tf.string)
#     z = tf.strings.join([x, y], name='join', separator=" ")
# # 开启会话，并执行计算图
# with tf.compat.v1.Session(graph=g) as sess:
#     result = sess.run(fetches=z, feed_dict={x: 'hello', y: 'world'})
#     print(result)

"""
    TensorFlow2.0构建动态计算图
"""
x = tf.constant('hello')
y = tf.constant('world')
z = tf.strings.join([x, y], separator=' ')
tf.print(z)


@tf.function
def strjoin(x, y):
    z = tf.strings.join([x, y], separator=' ')
    return z


# result = strjoin(tf.constant('hello'), tf.constant('world'))
# print(result)


import datetime
import os
from pathlib import Path
import tensorboard
# 创建日志
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = str(Path('./data/autograph/' + stamp))

writer = tf.summary.create_file_writer(logdir)
# 开启Autograph跟踪
tf.summary.trace_on(graph=True, profiler=True)
# 执行Autograph
result = strjoin('hello', 'world')
# 将计算图信息写入日志
with writer.as_default():
    tf.summary.trace_export(
        name='autograph',
        step=0,
        profiler_outdir=logdir
    )