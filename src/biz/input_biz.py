import sys, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.log_util import *
from data_handler.mapping_parser import MappingParser
import tensorflow as tf

FIELD_OUTER_DELIM = ' '
FIELD_INNER_DELIM = ' '
TYPE_MAPPING = {'string': tf.string, 'int': tf.int32, 'float': tf.float32}


def _parse_data(mp, value):
    features = dict()
    # features['sparse_col'] = dict()
    feature_num = len(mp.feature_list) + 1
    str_columns = tf.decode_csv(value, [['']] * feature_num, field_delim=FIELD_OUTER_DELIM)

    # column shape is (1,)
    for idx, column in enumerate(str_columns):

        if idx == 0:
            feature_name = 'label'
            feature_type = 'float'
        else:
            feature_name = mp.feature_list[idx - 1]
            feature_type = mp.feature_info[feature_name]['type']

        if feature_name in mp.skip_features:
            continue

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


def input_fn(mp: MappingParser, data_dir, epoch_num, batch_size, shuffle=False, return_iterator=False,
             drop_remainder=False):
    # data parse for one batch

    # get all data at data_dir
    data_files = tf.gfile.Glob(data_dir)
    data_set = tf.data.TextLineDataset(data_files)

    # dataset handle
    if shuffle:
        data_set = data_set.shuffle(buffer_size=batch_size * 3)
    data_set = data_set.batch(batch_size, drop_remainder=drop_remainder)
    data_set = data_set.map(lambda x: _parse_data(mp, x), num_parallel_calls=2)
    data_set = data_set.repeat(epoch_num)
    # data_set = data_set.interleave(
    #    lambda x, y: tf.data.Dataset.from_tensors((x, y)).repeat(epoch_num),
    #    cycle_length=16,
    #    block_length=batch_size)

    data_set = data_set.prefetch(1000)

    # return features and labels for input_fn
    if not return_iterator:
        iterator = data_set.make_one_shot_iterator()
        features, labels = iterator.get_next("iterator")
        return features, labels
    else:
        iterator = data_set.make_initializable_iterator()
        features, labels = iterator.get_next("iterator")
        return iterator, features, labels
