# 描述
感知机是一个简单的算法，给定n维向量x作为输入，通常称做输入特征或简单特征，输出为1（是）或0（否）。

![感知机公式](../../../base/img/感知机公式.png)

这里，w是权重向量，wx是点积

![点击公式](../../../base/img/点积公式.png)

b是偏差。```wx+b```定义了一个边界超平面，可以通过设置w和b的值来改变它的位置。如果x位于直线之上，则结果为正，否则为负。感知机不能表示非确定性答案。
# 训练感知机
 Keras的原始构造模块是模型，最简单的模型称为序贯模型，Keras的序贯模型是神经网络层的线性管道(堆栈)。以下代码段定义了一个包含12个人工神经元的单层网络，它预计有8个输入变量（特征）

```python
from keras.models import Sequential
from keras.layers import Dense
perceptron_model = Sequential()
perceptron_model.add(Dense(12, input_dim=8, kernel_initializer='random_uniform'))
```
 每个神经元可以用特定的权重进行初始化。Keras提供几个选择，其中常用的选择如下：
 * random_uniform：初始化权重为(-0.05, 0.05)之间的均匀随机的微小数值。换句话说，给定区间里的任何值都可能作为权重。
 * random_normal：根据高斯分布初始化权重，平均值为0，标准差为0.05。
 * zero：所有权重初始化为0。
