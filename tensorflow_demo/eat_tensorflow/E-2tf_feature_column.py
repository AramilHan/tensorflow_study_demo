# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/2/6 15:26
@brief: 特征列feature-column
"""
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models


# 打印日志
def print_log(info):
    now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "==========" * 8 + "%s" % now_time)
    print(info + '...\n\n')


# =======================================
# 构建数据管道
# =======================================
print_log('step1: prepare dataset...')

df_train_raw = pd.read_csv('./data/titanic/train.csv')
df_test_raw = pd.read_csv('./data/titanic/test.csv')
df_raw = pd.concat([df_train_raw, df_test_raw])


def prepare_df_data(df_raw):
    df_data = df_raw.copy()
    df_data.columns = [x.lower() for x in df_data.columns]
    df_data = df_data.rename(columns={'survived': 'label'})
    df_data = df_data.drop(['passengerid', 'name'], axis=1)
    for col, dtype in dict(df_data.dtypes).items():
        # 判断是否包含缺失值
        if df_data[col].hasnans:
            # 添加标识是否缺失列
            df_data[col + '_nan'] = pd.isna(df_data[col]).astype('int32')
            # 填充
            if dtype not in [np.object, np.str, np.unicode]:
                df_data[col].fillna(df_data[col].mean(), inplace=True)
            else:
                df_data[col].fillna('', inplace=True)
    return df_data


df_data = prepare_df_data(df_raw)
df_train = df_data.iloc[0: len(df_train_raw), :]
df_test = df_data.iloc[len(df_train_raw):, :]


# 从dataframe导入数据
def df_to_dataset(df, shuffle=True, batch_size=32):
    df_data = df.copy()
    if 'label' not in df_data.columns:
        ds = tf.data.Dataset.from_tensor_slices(df_data.to_dict(orient='list'))
    else:
        labels = df_data.pop('label')
        ds = tf.data.Dataset.from_tensor_slices((df_data.to_dict(orient='list'), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df_data))
    ds = ds.batch(batch_size)
    return ds


ds_train = df_to_dataset(df_train)
ds_test = df_to_dataset(df_test)


# =======================================
# 定义特征列
# =======================================
print_log('step2: make feature columns...')

feature_columns = []

# 数值列
for col in ['age', 'fare', 'parch', 'sibsp'] + [c for c in df_data.columns if c.endswith('_nan')]:
    feature_columns.append(tf.feature_column.numeric_column(col))

# 分桶列
age = tf.feature_column.numeric_column('age')
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# 类别列
# 注意：所有的Categorical Column类型最终都要通过indicator_column转换成Dense Column类型才能传入模型
sex = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(key='sex', vocabulary_list=['male', 'female']))
feature_columns.append(sex)

pclass = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(key='pclass', vocabulary_list=[1, 2, 3]))
feature_columns.append(pclass)

ticket = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket('ticket', 3))
feature_columns.append(ticket)

embarked = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(key='embarked', vocabulary_list=['S', 'C', 'B']))
feature_columns.append(embarked)

# 嵌入列
cabin = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket('cabin', 32), 2)
feature_columns.append(cabin)

# 交叉列
pclass_cate = tf.feature_column.categorical_column_with_vocabulary_list(key='pclass', vocabulary_list=[1, 2, 3])
crossed_feature = tf.feature_column.indicator_column(tf.feature_column.crossed_column([age_buckets, pclass_cate], hash_bucket_size=15))
feature_columns.append(crossed_feature)

# =======================================
# 定义模型
# =======================================
print_log('step3: define model...')

tf.keras.backend.clear_session()
model = tf.keras.Sequential([
    layers.DenseFeatures(feature_columns),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')])

# =======================================
# 训练模型
# =======================================
print_log('step4: train model...')

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history = model.fit(ds_train, validation_data=ds_test, epochs=10)

# =======================================
# 评估模型
# =======================================
print_log('step5: eval model...')
model.summary()


def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro--')
    plt.title('Training and validation' + metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend(['train_'+metric, 'val_'+metric])
    plt.show()


plot_metric(history, 'accuracy')
