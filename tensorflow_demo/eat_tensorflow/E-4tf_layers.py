# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/2/7 11:18
@brief: 模型层layers
"""
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# mypower = layers.Lambda(lambda x: tf.math.pow(x, 2))
# tf.print(mypower(tf.range(5)))


class Linear(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    # build方法一般定义layer需要被训练的参数
    def build(self, input_shape):
        self.w = self.add_weight('w', shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight('b', shape=(self.units, ), initializer='random_normal', trainable=True)
        super(Linear, self).build(input_shape)

    # call方法一般定义正向传播运算逻辑，__call__方法调用了它
    @tf.function
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    # 如果要让自定义layer通过Function API组合成模型时可以被保存成h5模型，需要自定义get_config方法
    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config


linear = Linear(units=8)
print(linear.built)
linear.build(input_shape=(None, 16))
print(linear.built)
