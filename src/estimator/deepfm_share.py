from tensorflow.python.feature_column.feature_column import input_layer
from .base_model import BaseModel
from tensorflow.python.ops import variable_scope
import tensorflow as tf
from tensorflow.python.ops.losses import losses

class DeepFMS(BaseModel):
    def __init__(self, **params):
        super(DeepFMS, self).__init__(**params)

    def _forward(self, features, is_training):
        embed_column = self.columns_dict.get('embed_fc')
        cate_column = self.columns_dict.get('indi_fc')
        if not embed_column:
            raise Exception('Need embed_fc given')

        ##### fm innerproduct part
        fm_cross_logits = 0
        feature_size = len(embed_column)
        embed_column.sort(key=lambda x: x.name)
        embed_tensors = [tf.feature_column.input_layer(features = {embed_column_i.name.split('_')[0]:features[embed_column_i.name.split('_')[0]]}, feature_columns = embed_column_i) for embed_column_i in embed_column]
        for feature_id_x in range(0,feature_size):
            for feature_id_y in range(feature_id_x + 1, feature_size):
                feature_tensor_x = embed_tensors[feature_id_x]
                feature_tensor_y = embed_tensors[feature_id_y]
                #print(feature_tensor_x.shape)
                #print(feature_tensor_y.shape)
                feature_multiply = tf.multiply(feature_tensor_x, feature_tensor_y)
                cross_sum = tf.reduce_sum( feature_multiply,1,keepdims=True)
                #print(cross_sum.shape)
                fm_cross_logits += cross_sum
                
                #print(feature_multiply.shape)
                #print(fm_logits.shape)

        ##### fm first-order part
        #cate_tensors = [tf.feature_column.input_layer(features = {cate_column_i.name.split('_')[0]:features[cate_column_i.name.split('_')[0]]}, feature_columns = cate_column_i) for cate_column_i in cate_column]
        #cate_concat = tf.concat(cate_tensors,1)
        #print(cate_concat.shape)
        #fm_first_logits = tf.layers.dense(cate_concat, units=1, use_bias = True, activation=None, kernel_initializer=self.kernel_initializer, name='fm_first_logits')
        #fm_logits = fm_first_logits + fm_cross_logits
        fm_logits = fm_cross_logits
       
        ##### deep part
        #deep_net = tf.feature_column.input_layer(features = features, feature_columns=embed_column)
        deep_net = tf.concat(embed_tensors,1)
        #print(deep_net.shape)
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
        #print(deep_logits.shape)
        #print(fm_logits.shape)
        #print(fm_logits)
        all_logits = fm_logits  + deep_logits
        #all_logits = fm_logits
		#print(all_logits.shape) 
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
    
