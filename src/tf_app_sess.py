import tensorflow as tf


# 随机生成 100 点（x，y）, [0,1）上的均匀分布 作为训练集和测试集
# import numpy as np
# x_data_sample = np.random.rand(10).astype(np.float32)
# y_data_sample = x_data_sample * 0.1 + 0.3
# for i in range(len(x_data_sample)):
#     print(x_data_sample[i], y_data_sample[i])

def log_local(_sess, _step, _x_data, _y_data, tag):
    print(tag, _step, _sess.run(W),
          _sess.run(b),
          _x_data,
          _y_data,
          sess.run(y, feed_dict={x_data: _x_data,
                                 y_data: _y_data}),
          sess.run(loss, feed_dict={x_data: _x_data,
                                    y_data: _y_data}))


def save_summary():
    if step % log_train_step == 0:
        log_local(sess, step, x_data_train, y_data_train, 'train')
        summary_train = sess.run(merged, feed_dict={x_data: x_data_train, y_data: y_data_train})
        write_train.add_summary(summary_train, step)
    if step % log_eval_step == 0:
        _x_data_eval, _y_data_eval = sess.run(next_element_eval)
        log_local(sess, step, _x_data_eval, _y_data_eval, 'eval')
        summary_eval = sess.run(merged, feed_dict={x_data: _x_data_eval, y_data: _y_data_eval})
        write_eval.add_summary(summary_eval, step)
    if checkpoint_step % 5 == 0:
        saver.save(sess, model_dir + '/model.ckpt', global_step=step,
                   write_meta_graph=True)


def input_fn(data_path, epoch_num, batch_size, prefetch_num):
    def _map_function(value: str):
        cols = tf.decode_csv(value, [0.0] * 2, field_delim=' ')
        tf.print('data_path:{},epoch_num:{},batch_size:{}'.format(data_path, epoch_num, batch_size))
        tf.print(cols)
        return cols[0], cols[1]

    # .shuffle(buffer_size=10 * 3) \
    data_set = tf.data.TextLineDataset(
        tf.gfile.Glob(data_path)) \
        .batch(batch_size, drop_remainder=True) \
        .map(_map_function) \
        .repeat(epoch_num) \
        .prefetch(prefetch_num)
    return data_set.make_initializable_iterator()


iterator_train = input_fn(data_path='/Users/yiche/dev/code/deep_model/data/esmm_block/train_data.txt',
                          batch_size=2, epoch_num=40, prefetch_num=5)
iterator_eval = input_fn(data_path='/Users/yiche/dev/code/deep_model/data/esmm_block/eval_data.txt',
                         batch_size=10, epoch_num=None, prefetch_num=1)
next_element_train = iterator_train.get_next()
next_element_eval = iterator_eval.get_next()

x_data = tf.placeholder(tf.float32, [None])
y_data = tf.placeholder(tf.float32, [None])
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
loss = tf.reduce_mean(tf.square(y - y_data))

tf.summary.scalar("loss", loss)
tf.summary.scalar("W", tf.reshape(W, []))
tf.summary.scalar("b", tf.reshape(b, []))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train_W = optimizer.minimize(loss, var_list=[W])
train_b = optimizer.minimize(loss, var_list=[b])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(iterator_train.initializer)
sess.run(iterator_eval.initializer)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=2500)
merged = tf.summary.merge_all()

model_dir = '/Users/yiche/dev/code/deep_model/model/esmm_block'
write_train = tf.summary.FileWriter(model_dir, sess.graph)
write_eval = tf.summary.FileWriter(model_dir + '/eval')

log_train_step = 1
log_eval_step = 10
checkpoint_step = 5
step = 0
while True:
    try:
        x_data_train, y_data_train = sess.run(next_element_train)  # 注意：要一次性读取出x、y，因为sess.run(x)时也会把y取出
    except tf.errors.OutOfRangeError:
        break
    # 记录过程信息
    save_summary()
    # update model variable
    if step % 2 == 0:
        sess.run(train_W, feed_dict={x_data: x_data_train, y_data: y_data_train})
    else:
        sess.run(train_b, feed_dict={x_data: x_data_train, y_data: y_data_train})
    step = step + 1

write_train.close()
write_eval.close()
sess.close()
