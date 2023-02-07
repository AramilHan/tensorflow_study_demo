# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/2/7 17:14
@brief: 回调函数callbacks
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, losses, metrics, callbacks
import tensorflow.keras.backend as k
import json


# json_log = open('./data/keras_log.json', mode='wt', buffering=1)
# json_logging_callback = callbacks.LambdaCallback(
#     on_epoch_end=lambda epoch, logs: json_log.write(json.dumps(dict(epoch=epoch, **logs)) + '\n'),
#     on_train_end=lambda logs: json_log.close()
# )


class LearingRateScheduler(callbacks.Callback):
    def __init__(self, schedule, verbose=0):
        super(LearingRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:
            lr = float(K.get_value(self.model.optimizer.lr))
            lr = self.schedule(epoch, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        if isinstance(lr, ops.Tensor) and not lr.dtype.is_floating:
            raise ValueError('The dtype of Tensor should be float')
        K.set_value(self.model.optimizer.lr, K.get_value(lr))
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
