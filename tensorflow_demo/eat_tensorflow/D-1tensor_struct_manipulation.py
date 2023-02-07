# -*- encoding: utf-8 -*-
"""
@author: aramil
@date: 2023/2/2 16:05
@brief: 张量的结构操作
"""
import tensorflow as tf
import numpy as np

"""
    创建张量
"""
a = tf.constant([1, 2, 3], dtype=tf.float32)
tf.print(a)
b = tf.range(1, 10, delta=2)
tf.print(b)
# 均匀分布
c = tf.linspace(0.0, 2 * 3.14, 100)
tf.print(c)
d = tf.zeros([3, 3])
tf.print(d)
a = tf.ones([3, 3])
tf.print(a)
b = tf.zeros_like(a, dtype=tf.float32)
tf.print(b)
b = tf.fill([3, 2], 5)
tf.print(b)
# 均匀分布随机
tf.random.set_seed(1.0)
a = tf.random.uniform([5], minval=0, maxval=10)
tf.print(a)
# 正态分布随机
b = tf.random.normal([3, 3], mean=0.0, stddev=1.0)
tf.print(b)
# 正态分布随机，剔除2倍方差以外数据重新生成
c = tf.random.truncated_normal((5, 5), mean=0.0, stddev=1.0, dtype=tf.float32)
tf.print(c)
# 特殊矩阵
I = tf.eye(3, 3)
tf.print(I)
tf.print(" ")
t = tf.linalg.diag([1, 2, 3])
tf.print(t)

"""
    索引切片
"""
tf.random.set_seed(3)
t = tf.random.uniform([5, 5], minval=0, maxval=10, dtype=tf.int32)
tf.print(t)

# 第0行
tf.print(t[0])
# 倒数第一行
tf.print(t[-1])
# 第1行第3列
tf.print(t[1, 3])
tf.print(t[1][3])
# 第1行至第3行
tf.print(t[1:4, :])
tf.print(tf.slice(t, [1, 0], [3, 5]))
# 第1行至最后一行，第0列到最后1列每隔两列取1列
tf.print(t[1: 4, :4:2])
# 对变量来说，还可以使用索引和切片修改部分元素
x = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)
x[1, :].assign(tf.constant([0.0, 0.0]))
tf.print(x)
a = tf.random.uniform([3, 3, 3], minval=0, maxval=10, dtype=tf.int32)
tf.print(a)
# 省略号可以表示多个冒号
tf.print(a[..., 1])

tf.print("==" * 4, "不规则切片", "==" * 4)
# 班级成绩册，4个班级，10个学生，7门科目成绩。即4x10x7张量
scores = tf.random.uniform((4, 10, 7), minval=0, maxval=100, dtype=tf.int32)
tf.print(scores)
# 抽取每个班第0个学生，第5个学生，第9个学生的全部成绩
tf.print("抽取每个班第0个学生，第5个学生，第9个学生的全部成绩")
p = tf.gather(scores, [0, 5, 9], axis=1)
tf.print(p)
# 抽取每个班第0个学生，第5个学生，第9个学生的第1门课程，第3门课程，第6门课程成绩
tf.print("抽取每个班第0个学生，第5个学生，第9个学生的第1门课程，第3门课程，第6门课程成绩")
q = tf.gather(tf.gather(scores, [0, 5, 9], axis=1), [1, 3, 6], axis=2)
tf.print(q)
# 抽取第0个班第0个学生，第2个班第4个学生，第3个班第6个学生的全部成绩
# indices的长度为采样样本的个数，每个元素为采样位置的坐标
tf.print("抽取第0个班第0个学生，第2个班第4个学生，第3个班第6个学生的全部成绩")
s = tf.gather_nd(scores, indices=[(0, 0), (2, 4), (3, 6)])
tf.print(s)
# 抽取每个班第0个学生，第5个学生，第9个学生的全部成绩
tf.print("抽取每个班第0个学生，第5个学生，第9个学生的全部成绩")
p = tf.boolean_mask(scores, [True, False, False, False, False,
                             True, False, False, True, False], axis=1)
tf.print(p)
# 抽取第0个班第0个学生，第2个班第4个学生，第3个班第6个学生的全部成绩
tf.print("抽取第0个班第0个学生，第2个班第4个学生，第3个班第6个学生的全部成绩")
s = tf.boolean_mask(scores, [[True, False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False, False, False, False],
                             [False, False, False, False, True, False, False, False, False, False],
                             [False, False, False, False, False, False, True, False, False, False]])
tf.print(s)
# 利用tf.boolean_mask可以实现布尔索引
# 找到矩阵中小于0的元素
tf.print("找到矩阵中小于0的元素")
c = tf.constant([[-1, 1, -1], [2, 2, -2], [3, -3, 3]], dtype=tf.float32)
tf.print(c, '\n')
tf.print(tf.boolean_mask(c, c < 0), '\n')
tf.print(c[c < 0])  # 等价于boolean_mask

tf.print("==" * 4, "通过修改张量获取新的张量", "==" * 4)
# 找到张量中小于0的元素，将其换成np.nan得到新的张量
tf.print("找到张量中小于0的元素，将其换成np.nan得到新的张量")
c = tf.constant([[-1, 1, -1], [2, 2, -2], [3, -3, 3]], dtype=tf.float32)
d = tf.where(c < 0, tf.fill(c.shape, np.nan), c)
tf.print(d)
# 如果where只有一个参数，将返回所有满足条件的位置坐标
tf.print("如果where只有一个参数，将返回所有满足条件的位置坐标")
indices = tf.where(c < 0)
tf.print(indices)
# 将张量的第[0, 0]和[2, 1]两个位置元素替换为0得到新的张量
tf.print("将张量的第[0, 0]和[2, 1]两个位置元素替换为0得到新的张量")
d = c - tf.scatter_nd([[0, 0], [2, 1]], [c[0, 0], c[2, 1]], c.shape)
tf.print(d)
# scatter_nd的作用和gather_nd有些相反
# 可以将某些值插入到一个给定shape的全0的张量的指定位置处
tf.print("可以将某些值插入到一个给定shape的全0的张量的指定位置处")
indices = tf.where(c < 0)
tf.print(indices)
d = tf.scatter_nd(indices, tf.gather_nd(c, indices), c.shape)
tf.print(d)

tf.print("==" * 4, "维度变换", "==" * 4)
a = tf.random.uniform(shape=[1, 3, 3, 2], minval=0, maxval=255, dtype=tf.int32)
tf.print(a.shape)
tf.print(a)
# 改成(3,6)形状的张量
tf.print("改成[3,6]形状的张量")
b = tf.reshape(a, [3, 6])
tf.print(b.shape)
tf.print(b)
# 改回[1, 3, 3, 2]形状的张量
tf.print("改回[1, 3, 3, 2]形状的张量")
c = tf.reshape(b, [1, 3, 3, 2])
tf.print(c.shape)
tf.print(c)

s = tf.squeeze(a)
tf.print(s.shape)
tf.print(s)

tf.print("=" * 4, "在第0维度插入长度为1的一个维度", "=" * 4)
d = tf.expand_dims(s, axis=0)
tf.print(d)

tf.print("=" * 4, "Batch，Height，Width，Channel", "=" * 4)
a = tf.random.uniform(shape=[100, 600, 600, 4], minval=0, maxval=255, dtype=tf.int32)
tf.print(a.shape)
tf.print("=" * 4, "转换成Channel，Height，Width，Batch", "=" * 4)
s = tf.transpose(a, perm=[3, 1, 2, 0])
tf.print(s.shape)

a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
c = tf.constant([[9.0, 10.0], [11.0, 12.0]])
tf.print("=" * 4, "tf.concat", "=" * 4)
tf.print(tf.concat([a, b, c], axis=0))
tf.print(tf.concat([a, b, c], axis=1))
tf.print("=" * 4, "tf.stack", "=" * 4)
tf.print(tf.stack([a, b, c], axis=0))
tf.print(tf.stack([a, b, c], axis=1))

a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
c = tf.constant([[9.0, 10.0], [11.0, 12.0]])

c = tf.concat([a, b, c], axis=0)
print("="*4, "指定分割份数，平均分割", "="*4)
print(tf.split(c, 3, axis=0))
print("="*4, "指定每份的记录数量", "="*4)
print(tf.split(c, [2, 2, 2], axis=0))
