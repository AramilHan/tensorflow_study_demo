# -*- encoding: utf-8 -*-
"""
@author:
@date: 2022/12/27 19:17
@brief:
"""
import keras.callbacks
from keras import regularizers
from keras.models import Sequential
from keras.layers.core import Dense

model = Sequential()
model.add(Dense(64, input_dim=64, W_regularizer=regularizers.l2()))
json_string = model.to_json()
yaml_string = model.to_yaml()

from keras.models import model_from_json, model_from_yaml
model = model_from_json(json_string)
model = model_from_yaml(yaml_string)

from keras.models import load_model
model.save('my_model.h5')
load_model('my_model.h5')

keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0
)

