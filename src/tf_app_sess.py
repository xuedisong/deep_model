import tensorflow as tf

import numpy as np

# 随机生成 100 点（x，y）
# [0,1）上的均匀分布
x_data = np.random.rand(10).astype(np.float32)

y_data = x_data * 0.1 + 0.3

# 构建线性模型的 tensor 变量 W, b

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

b = tf.Variable(tf.zeros([1]))

y = W * x_data + b

# 构建损失方程，优化器及训练模型操作 train

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.001) # 0.5

train = optimizer.minimize(loss)

# 构建变量初始化操作 init

init = tf.initialize_all_variables()

# 构建 TensorFlow session

sess = tf.Session()

# 初始化所有 TensorFlow 变量

sess.run(init)

# 训练该线性模型，每隔 20 次迭代，输出模型参数

for step in range(201):
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b), x_data, sess.run(y))
    sess.run(train)
sess.close()
