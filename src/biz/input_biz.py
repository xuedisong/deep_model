from typing import List

import tensorflow as tf
from tensorflow.python.data.ops.readers import TextLineDatasetV1

from bean.feature import Feature
from utils.log_util import *

FIELD_OUTER_DELIM = ' '
FIELD_INNER_DELIM = ' '
TYPE_MAPPING = {'string': tf.string, 'int': tf.int32, 'float': tf.float32}


def _to_DenseTensor(record, featureList: List[Feature]):
    features = dict()
    # features['sparse_col'] = dict()
    feature_num = len(featureList) + 1
    str_columns = tf.decode_csv(record, [['']] * feature_num, field_delim=FIELD_OUTER_DELIM)

    # column shape is (1,)
    for idx, column in enumerate(str_columns):
        if idx == 0:
            feature_name = 'label'
            feature_type = 'float'
        else:
            feature = featureList[idx - 1]
            feature_name = feature.get_name()
            feature_type = feature.get_type()

        # UNDER multi_val MODE, dense shape of SparseTensor sparse_col is [batch_size, column_split_size]
        # different feature with different column split size has different feature size

        sparse_col = tf.string_split(column, delimiter=FIELD_INNER_DELIM)
        if feature_type == 'string':
            dense_col = tf.sparse.to_dense(sparse_col, "")
        elif feature_type == 'int':
            values = tf.string_to_number(sparse_col.values, tf.int32)
            dense_col = tf.sparse_to_dense(sparse_col.indices, sparse_col.dense_shape, values, 0)
        elif feature_type == "float":
            values = tf.string_to_number(sparse_col.values, tf.float32)
            dense_col = tf.sparse_to_dense(sparse_col.indices, sparse_col.dense_shape, values, 0.0)
        else:
            err_log('Unrecognized feature type: ' + feature_type)
            continue
        features[feature_name] = dense_col
    labels = features.pop('label')
    return features, labels


def input_fn(featureList: List[Feature], data_dir, epoch_num, batch_size, shuffle=False, return_iterator=False,
             drop_remainder=False):
    # data parse for one batch

    # get all data at data_dir
    data_files = tf.gfile.Glob(data_dir)
    data_set: TextLineDatasetV1 = tf.data.TextLineDataset(data_files)

    # dataset handle
    if shuffle:
        data_set = data_set.shuffle(buffer_size=batch_size * 3)
    data_set = data_set.batch(batch_size, drop_remainder=drop_remainder) \
        .map(lambda record: _to_DenseTensor(record, featureList), num_parallel_calls=2) \
        .repeat(epoch_num) \
        .prefetch(1000)
    # data_set = data_set.interleave(
    #    lambda x, y: tf.data.Dataset.from_tensors((x, y)).repeat(epoch_num),
    #    cycle_length=16,
    #    block_length=batch_size)

    # return features and labels for input_fn
    if return_iterator:
        iterator = data_set.make_initializable_iterator()
        features, labels = iterator.get_next("iterator")
        return iterator, features, labels
    else:
        iterator = data_set.make_one_shot_iterator()
        features, labels = iterator.get_next("iterator")
        return features, labels


if __name__ == '__main__':
    import tensorflow as tf
    from common import context
    from utils.args_util import FLAGS

    iterator, features, labels = input_fn(context.featureList, FLAGS.train_data, epoch_num=FLAGS.train_epochs,
                                          batch_size=100,
                                          shuffle=FLAGS.shuffle, return_iterator=True, drop_remainder=True)

    data_iter = iterator.get_next()  # 和 iterator.get_next("iterator") 相同

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)

    count = 0
    while True:
        try:
            feature, label = sess.run(data_iter)  # data_iter 是个(tensor,tensor)元组，需要tf.sess运行才知道其实际的形状及值。
            # print(count)
            # print(tf.get_collection('parse_value'))
            # print(tf.get_collection('str_columns'))
            for k, v in feature.items():
                print(k)
                print(v.shape)
                print(v)
                break
            print(label.shape)
            print(label)
            count += 1
        except tf.errors.OutOfRangeError:
            print("break")
            break
    print("end")
