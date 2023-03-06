import tensorflow as tf
import numpy as np


# from sklearn.metrics import roc_auc_score

def roc_auc_score(labels, preds, n_bins=100):
    postive_len = sum(labels)
    negative_len = len(labels) - postive_len
    total_case = postive_len * negative_len
    pos_histogram = [0 for _ in range(n_bins)]
    neg_histogram = [0 for _ in range(n_bins)]
    bin_width = 1.0 / n_bins
    for i in range(len(labels)):
        nth_bin = int(preds[i] / bin_width)
        if labels[i] == 1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
        accumulated_neg += neg_histogram[i]

    return satisfied_pair / float(total_case)


y_true = [1, 1, 0, 0, 1, 1, 0]
y_pred = [0.8, 0.7, 0.5, 0.5, 0.5, 0.5, 0.3]
print(roc_auc_score(y_true, y_pred))


# 随机生成 100 点（x，y）, [0,1）上的均匀分布 作为训练集和测试集
# import numpy as np
# x_data_sample = np.random.rand(10).astype(np.float32)
# y_data_sample = x_data_sample * 0.1 + 0.3
# for i in range(len(x_data_sample)):
#     print(x_data_sample[i], y_data_sample[i])

def log_local(_sess, _step, _feed_dict, tag):
    _y_pred = sess.run(y, feed_dict=_feed_dict)
    _y_true = [1 if i > 0.35 else 0 for i in _feed_dict[labels['y_data']]]
    print(tag, _step, _sess.run(W),
          _sess.run(b),
          _feed_dict[features['x_data']],
          _feed_dict[labels['y_data']],
          sess.run(y, feed_dict=_feed_dict),
          sess.run(loss, feed_dict=_feed_dict),
          roc_auc_score(_y_true, _y_pred))


def save_summary():
    if step % log_train_step == 0:
        log_local(sess, step, feed_dict_train, 'train')
        summary_train = sess.run(merged, feed_dict=feed_dict_train)
        write_train.add_summary(summary_train, step)
    if step % log_eval_step == 0:
        _features_eval, _labels_eval = sess.run(next_element_eval)
        _feed_dict_eval = {features[k]: v for k, v in _features_eval.items()}
        for k, v in _labels_eval.items():
            _feed_dict_eval[labels[k]] = v
        log_local(sess, step, _feed_dict_eval, 'eval')
        summary_eval = sess.run(merged, feed_dict=_feed_dict_eval)
        write_eval.add_summary(summary_eval, step)
    if step % checkpoint_step == 0:
        saver.save(sess, model_dir + '/model.ckpt', global_step=step,
                   write_meta_graph=True)


def input_fn(data_path, epoch_num, batch_size, prefetch_num):
    def _map_function(value: str):
        cols = tf.decode_csv(value, [0.0] * 2, field_delim=' ')
        tf.print('data_path:{},epoch_num:{},batch_size:{}'.format(data_path, epoch_num, batch_size))
        tf.print(cols)
        return {'x_data': cols[0], 'x_data2': cols[0]}, {'y_data': cols[1], 'y_data2': cols[1]}

    # .shuffle(buffer_size=10 * 3) \
    data_set = tf.data.TextLineDataset(
        tf.gfile.Glob(data_path)) \
        .batch(batch_size, drop_remainder=True) \
        .map(_map_function) \
        .repeat(epoch_num) \
        .prefetch(prefetch_num)
    return data_set.make_initializable_iterator()


iterator_train = input_fn(data_path='/Users/yiche/dev/code/deep_model/data/esmm_block/train_data.txt',
                          batch_size=10, epoch_num=8, prefetch_num=5)
iterator_eval = input_fn(data_path='/Users/yiche/dev/code/deep_model/data/esmm_block/eval_data.txt',
                         batch_size=10, epoch_num=None, prefetch_num=1)
next_element_train = iterator_train.get_next()
next_element_eval = iterator_eval.get_next()

features = {'x_data': tf.placeholder(tf.float32, [None], name='x_data_ph'),
            'x_data2': tf.placeholder(tf.float32, [None], name='x_data2_ph')}
labels = {'y_data': tf.placeholder(tf.float32, [None], name='y_data_ph'),
          'y_data2': tf.placeholder(tf.float32, [None], name='y_data2_ph')}

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * features['x_data'] + b
loss = tf.reduce_mean(tf.square(y - labels['y_data']))

tf.summary.scalar("loss", loss)
tf.summary.scalar("W", tf.reshape(W, []))
tf.summary.scalar("b", tf.reshape(b, []))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train_W = optimizer.minimize(loss, var_list=[W])
train_b = optimizer.minimize(loss, var_list=[b])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator_train.initializer)
    sess.run(iterator_eval.initializer)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=2500)
    merged = tf.summary.merge_all()

    model_dir = '/Users/yiche/dev/code/deep_model/model/esmm_block'
    log_train_step = 1
    log_eval_step = 2
    checkpoint_step = 5
    step = 0
    with tf.summary.FileWriter(model_dir, sess.graph) as write_train:
        with tf.summary.FileWriter(model_dir + '/eval') as write_eval:
            while True:
                try:
                    features_train, labels_train = sess.run(
                        next_element_train)  # 注意：要一次性读取出x、y，因为sess.run(x)时也会把y取出
                    feed_dict_train = {features[k]: v for k, v in features_train.items()}
                    for k, v in labels_train.items():
                        feed_dict_train[labels[k]] = v
                except tf.errors.OutOfRangeError:
                    break
                # 记录过程信息
                save_summary()
                # update model variable
                if step % 2 == 0:
                    sess.run(train_W, feed_dict=feed_dict_train)
                else:
                    sess.run(train_b, feed_dict=feed_dict_train)
                step = step + 1
