# -*- encoding: utf-8 -*-
"""
@author:
@date: 2022/9/21 11:25
@brief:
"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
hello = tf.constant('Hello TensorFlow!')
sess = tf.compat.v1.Session()
print(sess.run(hello))
