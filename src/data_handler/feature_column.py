import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.log_util import *

from .mapping_parser import MappingParser
import tensorflow as tf


class FeatureColumn(object):
    def __init__(self, mp):
        if not isinstance(mp, MappingParser):
            raise TypeError('Given mp should be object of MappingParser')
        if not mp.feature_info:
            raise Exception('Feature info has not been initialized')
        self.mp = mp

    def build_column(self, embedding_size, combiner=None, max_norm=None):
        column_result = dict()
        for feature_name, info in self.mp.feature_info.items():
            if feature_name in self.mp.skip_features:
                continue
            if info['type'] == 'string':
                if feature_name not in self.mp.vocabulary_map: 
                    continue
                fc = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, self.mp.vocabulary_map[feature_name])
                #t = tf.glorot_uniform_initializer()
                #fc_emb = tf.feature_column.embedding_column(fc, dimension=embedding_size,initializer = t)
                fc_emb = tf.feature_column.embedding_column(fc, dimension=embedding_size,initializer = tf.glorot_uniform_initializer)
                fc_indi = tf.feature_column.indicator_column(fc)
                column_result.setdefault('embed_fc', []).append(fc_emb)
                column_result.setdefault('indi_fc',[]).append(fc_indi)
            elif info['type'] == 'int':
                fc = tf.feature_column.numeric_column(feature_name, shape=1, default_value=0, dtype=tf.int64)
                column_result.setdefault('value_fc', []).append(fc)
            elif info['type'] == 'float':
                fc = tf.feature_column.numeric_column(feature_name, shape=1, default_value=0.0, dtype=tf.float32)
                column_result.setdefault('value_fc', []).append(fc)
            else:
                err_log('Warning! Wrong feature type {}!'.format(info['type']))
                continue

        return column_result


# test build feature column
if __name__ == '__main__':
	feature_info_file = '../../data/data_demo/feature_info.txt' 
	dict_info_file = '../../data/data_demo/dict.txt'
	data_dir = '../../data/data_demo/train_demo.txt'
	mp = MappingParser(feature_info_file, dict_info_file)

	fc = FeatureColumn(mp)
	result = fc.build_column(16)
	for k, v in result.items():
		print(k)
		for i in v:
			print(type(i))

