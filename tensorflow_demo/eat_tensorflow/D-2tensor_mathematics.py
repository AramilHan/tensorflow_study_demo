# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/2/2 22:46
@brief: 张量的数学运算
"""
import tensorflow as tf

a = tf.range(1, 10)
tf.print(tf.reduce_sum(a))
tf.print(tf.reduce_mean(a))
tf.print(tf.reduce_max(a))
tf.print(tf.reduce_min(a))
tf.print(tf.reduce_prod(a))

b = tf.reshape(a, (3, 3))
tf.print(b)
tf.print(tf.reduce_sum(b, axis=1, keepdims=True))
tf.print(tf.reduce_sum(b, axis=0, keepdims=True))

p = tf.constant([True, False, False])
q = tf.constant([False, False, True])
tf.print(tf.reduce_all(p))
tf.print(tf.reduce_any(q))

s = tf.foldr(lambda a, b: a+b, tf.range(10))
tf.print(s)

a = tf.range(1, 10)
tf.print(tf.math.cumsum(a))
tf.print(tf.math.cumprod(a))

a = tf.range(1, 10)
tf.print(tf.argmax(a))
tf.print(tf.argmin(a))

a = tf.constant([1, 3, 7, 5, 4, 8])
values, indices = tf.math.top_k(a, 3, sorted=True)
tf.print(values)
tf.print(indices)

a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[2, 0], [0, 2]])
# tf.matmul(a,b)  <=>  a@b
tf.print(a@b)

a = tf.constant([[1, 2], [3, 4]])
tf.print(tf.transpose(a))

a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
tf.print(tf.linalg.inv(a))

a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
tf.print(tf.linalg.trace(a))

a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
tf.print(tf.linalg.norm(a))

a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
tf.print(tf.linalg.det(a))

a = tf.constant([[1, 2], [-5, 4]], dtype=tf.float32)
tf.print(tf.linalg.eigvals(a))

a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
q, r = tf.linalg.qr(a)
tf.print(q)
tf.print(r)
tf.print(q@r)

a = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
s, u, v = tf.linalg.svd(a)
tf.print(u, '\n')
tf.print(s, '\n')
tf.print(v, '\n')
tf.print(u@tf.linalg.diag(s)@tf.transpose(v))
