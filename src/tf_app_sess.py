import tensorflow as tf
import numpy as np

# 随机生成 100 点（x，y）, [0,1）上的均匀分布
x_data = tf.placeholder(tf.float32, [10, ])
y_data = tf.placeholder(tf.float32, [10, ])
x_data_train = np.random.rand(10).astype(np.float32)
y_data_train = x_data_train * 0.1 + 0.3
x_data_eval = np.random.rand(10).astype(np.float32)
y_data_eval = x_data_eval * 0.1 + 0.3

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
loss = tf.reduce_mean(tf.square(y - y_data))

tf.summary.scalar("loss", loss)

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables(), max_to_keep=2500)
merged = tf.summary.merge_all()

model_dir = '/Users/yiche/dev/code/deep_model/model/esmm_block'
write_train = tf.summary.FileWriter(model_dir, sess.graph)
write_eval = tf.summary.FileWriter(model_dir + '/eval', sess.graph)

for step in range(201):
    if step % 1 == 0:
        print(step, sess.run(W),
              sess.run(b),
              sess.run(loss, feed_dict={x_data: x_data_train, y_data: y_data_train}))  # x_data, sess.run(y)
        result_train = sess.run(merged, feed_dict={x_data: x_data_train, y_data: y_data_train})
        write_train.add_summary(result_train, step)
    if step % 10 == 0:
        result_eval = sess.run(merged, feed_dict={x_data: x_data_eval, y_data: y_data_eval})
        write_eval.add_summary(result_eval, step)
    if step % 5 == 0:
        saver.save(sess, '/Users/yiche/dev/code/deep_model/model/esmm_block' + '/model.ckpt', global_step=step,
                   write_meta_graph=True)
    # update model variable
    sess.run(train, feed_dict={x_data: x_data_train, y_data: y_data_train})
write_train.close()
write_eval.close()
sess.close()
