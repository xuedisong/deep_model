import tensorflow as tf
import numpy as np

# 随机生成 100 点（x，y）, [0,1）上的均匀分布
x_data = np.random.rand(10).astype(np.float32)
y_data = x_data * 0.1 + 0.3

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
loss = tf.reduce_mean(tf.square(y - y_data))

tf.summary.scalar("loss", loss)

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=2500)
merged = tf.summary.merge_all()
write = tf.summary.FileWriter('/Users/yiche/dev/code/deep_model/model/esmm_block', sess.graph)

for step in range(201):
    if step % 1 == 0:
        print(step, sess.run(W), sess.run(b), sess.run(loss))  # x_data, sess.run(y)
        result = sess.run(merged)
        write.add_summary(result, step)
    if step % 5 == 0:
        saver.save(sess, '/Users/yiche/dev/code/deep_model/model/esmm_block' + '/model.ckpt', global_step=step,
                   write_meta_graph=True)
    sess.run(train)
write.close()
sess.close()
