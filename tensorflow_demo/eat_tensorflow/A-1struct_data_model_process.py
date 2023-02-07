# -*- encoding: utf-8 -*-
"""
@author: Aramil
@date: 2023/1/18 15:29
@brief: 结构化数据建模流程范例
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers

"""
    设置pandas head展示
"""
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

train_raw_df = pd.read_csv('./data/titanic/train.csv')
test_raw_df = pd.read_csv('./data/titanic/test.csv')
# print(train_raw_df.head(10))

"""
    label分布情况
"""
# ax = train_raw_df['Survived'].value_counts().plot(kind='bar', figsize=(12, 8), fontsize=15, rot=0)
# ax.set_ylabel('Counts', fontsize=15)
# ax.set_xlabel('Survived', fontsize=15)
# plt.show()
"""
    年龄分布情况
"""
# ax = train_raw_df['Age'].plot(kind='hist', bins=20, color='purple', figsize=(12, 8), fontsize=15)
# ax.set_ylabel('Frequency', fontsize=15)
# ax.set_xlabel('Age', fontsize=15)
# plt.show()
"""
    年龄和label的相关性
"""
# ax = train_raw_df.query('Survived == 0')['Age'].plot(kind='density', figsize=(12, 8), fontsize=15)
# train_raw_df.query('Survived == 1')['Age'].plot(kind='density', figsize=(12, 8), fontsize=15)
# ax.legend(['Survived == 0', 'Survived == 1'], fontsize=12)
# ax.set_ylabel('Density', fontsize=15)
# ax.set_xlabel('Age', fontsize=15)
# plt.show()
"""
    数据预处理
"""


def preprocessing(df_data):
    df_result = pd.DataFrame()
    # Pclass
    df_Pclass = pd.get_dummies(df_data['Pclass'])  # get_dummies进行One-Hot编码
    df_Pclass.columns = ['Pclass_' + str(x) for x in df_Pclass.columns]
    df_result = pd.concat([df_result, df_Pclass], axis=1)

    # Sex
    df_sex = pd.get_dummies(df_data['Sex'])
    df_result = pd.concat([df_result, df_sex], axis=1)

    # Age
    df_result['Age'] = df_data['Age'].fillna(0)  # 补充缺失值
    df_result['Age_null'] = pd.isna(df_data['Age']).astype('int32')  # 补充年龄是否缺失特征

    # SibSp,Parch,Fare
    df_result['SibSp'] = df_data['SibSp']
    df_result['Parch'] = df_data['Parch']
    df_result['Fare'] = df_data['Fare']

    # Carbin
    df_result['Cabin_null'] = pd.isna(df_data['Cabin']).astype('int32')

    # Embarked
    df_Embarked = pd.get_dummies(df_data['Embarked'], dummy_na=True)
    df_Embarked.columns = ['Embarked_' + str(x) for x in df_Embarked.columns]
    df_result = pd.concat([df_result, df_Embarked], axis=1)
    return df_result


# 训练数据集特征列表
x_train = preprocessing(train_raw_df)
# 训练数据集label
y_train = train_raw_df['Survived'].values
# 测试数据集特征列表
x_test = preprocessing(test_raw_df)
# 测试数据集label
y_test = test_raw_df['Survived'].values
print('x_train.shape=', x_train.shape)
print('x_test.shape=', x_test.shape)
"""
    定义模型
"""
tf.keras.backend.clear_session()
model = models.Sequential()
model.add(layers.Dense(20, activation='relu', input_shape=(15,)))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
"""
    训练模型
"""
# 二分类问题选择二元交叉熵损失函数
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
history = model.fit(x_train, y_train, batch_size=64, epochs=30, validation_split=0.2)
"""
    评估模型
"""


# 评估模型训练集和验证集上的效果
def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro--')
    plt.title('Training and validation ' + metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend(['train_'+metric, 'val_'+metric])
    plt.show()


# plot_metric(history, 'loss')
# plot_metric(history, 'auc')
print('测试集评估')
model.evaluate(x=x_test, y=y_test)
"""
    使用模型
"""
# 预测概率
# predict_rate = model.predict(x_test[0:10])
# print(predict_rate)
# 预测类别
# predict_class = (model.predict(x_test[0:10]) > 0.5).astype('int32')
# print(predict_class)
"""
    保存模型
"""
# Keras方式保存
# model.save('./data/keras_model.h5')
# del model
# model = models.load_model('./data/keras_model.h5')
# model.evaluate(x_test, y_test)
# 保存模型结构
# json_str = model.to_json()
# 恢复模型结构
# model_json = models.model_from_json(json_str)
# 保存模型权重
# model.save_weights('./data/keras_model_weight.h5')
# 恢复模型结构
# model_json = models.model_from_json(json_str)
# model_json.compile(
#         optimizer='adam',
#         loss='binary_crossentropy',
#         metrics=['AUC']
#     )
# 加载权重
# model_json.load_weights('./data/keras_model_weight.h5')
# model_json.evaluate(x_test, y_test)

# Tensorflow原生方式保存
# 保存权重，该方式仅仅保存权重张量
model.save_weights('./data/tf_model_weights.ckpt', save_format='tf')
# 保存模型结构与模型参数到文件，该方式保存的模型具有跨平台性便于部署
model.save('./data/tf_model_savedmodel', save_format='tf')
print('export saved model')
model_loaded = tf.keras.models.load_model('./data/tf_model_savedmodel')
model_loaded.evaluate(x_test, y_test)
