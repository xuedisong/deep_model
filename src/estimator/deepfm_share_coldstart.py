from tensorflow.python.feature_column.feature_column import input_layer

from utils.stream_util import Stream
from .base_model import BaseModel
from tensorflow.python.ops import variable_scope
import tensorflow as tf
from tensorflow.python.ops.losses import losses
from common.context import featureList

class DeepFMSCS(BaseModel):
    def __init__(self, **params):
        super(DeepFMSCS, self).__init__(**params)

    def _forward(self, features, is_training):
        embed_column = self.columns_dict.get('embed_fc')
        cate_column = self.columns_dict.get('indi_fc')
        if not embed_column:
            raise Exception('Need embed_fc given')

        #### fm_part logits
        fm_cross_logits = 0
        feature_size = len(embed_column)
        embed_column.sort(key=lambda x: x.name)
        embed_tensors = []

        ##### coldstart embedding_lookup rebuild
        coldstart_names = ['itid','adplan','adgroup','ormodel','orbrand']
        for embed_col in embed_column:
            #### embedding_table build
            feature_name = embed_col.name.split('_')[0]
            vocabulary_list = Stream(featureList).filter(lambda feature:feature.get_name()==feature_name).find_first().get_valueList()
            table = tf.contrib.lookup.index_table_from_tensor(mapping = vocabulary_list, default_value=-1,num_oov_buckets=1)
            #t = tf.glorot_uniform_initializer()
            #t = tf.initializers.he_normal()
            t = tf.initializers.random_normal(mean=0, stddev=0.1)
            embed_matrix=tf.get_variable(name=feature_name+'_embmatrix',shape=[len(vocabulary_list),self.embedding_size],initializer=t,trainable=True)
            if feature_name in coldstart_names:
                oov_embed = tf.reduce_mean(embed_matrix,0)
            else:
                oov_embed = tf.zeros([self.embedding_size])
            #embed_all_matrix = tf.Variable(tf.concat([embed_matrix,[oov_embed]],0),name= feature_name + '_emball',trainable=True)
            embed_all_matrix = tf.concat([embed_matrix,[oov_embed]],0,name=feature_name + '_concat')

            tags = table.lookup(features[feature_name])
            idx = tf.where(tf.not_equal(tags,-1))
            gather_result = tf.gather_nd(tags,idx)
            sparse_tags = tf.SparseTensor(idx, gather_result, tf.shape(tags,out_type=tf.int64))
            embed_lookup = tf.nn.embedding_lookup_sparse(params=embed_all_matrix, sp_ids=sparse_tags, sp_weights=None, combiner= "mean")
            #embed_lookup = tf.nn.embedding_lookup(embed_all_matrix,ids=tags)
            #embed_layer = tf.reduce_mean(embed_lookup,1)
            #embed_lookup = tf.contrib.layers.safe_embedding_lookup_sparse(embed_all_matrix,sparse_ids=tags,combiner="mean")

            embed_tensors.append(embed_lookup)

                   
        for feature_id_x in range(0,feature_size):
            for feature_id_y in range(feature_id_x + 1, feature_size):
                feature_tensor_x = embed_tensors[feature_id_x]
                feature_tensor_y = embed_tensors[feature_id_y]
                feature_multiply = tf.multiply(feature_tensor_x, feature_tensor_y)
                cross_sum = tf.reduce_sum( feature_multiply,1,keepdims=True)
                fm_cross_logits += cross_sum
                
        fm_logits = fm_cross_logits


        #### deep_part logits
        deep_net = tf.concat(embed_tensors,1)
        for layer_id, units in enumerate(self.deep_hidden_units):
            with variable_scope.variable_scope('hiddenlayer_%d' % layer_id) as hidden_layer_scope:
                deep_net = tf.layers.dense(deep_net,units = units, activation=None, kernel_initializer=self.kernel_initializer, name=hidden_layer_scope)

                if self.batch_norm == 'on' and self.batch_norm_layers[layer_id] == 'on':
                    deep_net = tf.layers.batch_normalization(deep_net, training=is_training, name='bn_' + str(layer_id))

                if self.activation_func is not None:
                    deep_net = self.activation_func(deep_net)

                if self.dropout:
                    deep_net = tf.layers.dropout(deep_net, rate=self.dropout, training=is_training)

        ##### loss part
        deep_logits = tf.layers.dense(deep_net, units=1, activation=None, kernel_initializer=self.kernel_initializer, name='deep_logits')
        all_logits = fm_logits  + deep_logits
        return all_logits


    def get_model(self, features, labels, mode, params):
        if not self.deep_hidden_units:
            raise ValueError("Need deep_hidden_units given")

        is_training = False
        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True
        all_logits = self._forward(features, is_training)

        if mode != tf.estimator.ModeKeys.PREDICT:
            loss = losses.sigmoid_cross_entropy(labels, all_logits)

        else:
            loss = None

        return self.get_estimator_spec(mode, all_logits, loss, labels)
    
