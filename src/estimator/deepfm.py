from .base_model import BaseModel
import tensorflow as tf
from tensorflow.python.ops import variable_scope

from .base_model import BaseModel


class DeepFM(BaseModel):
    def __init__(self, **params):
        super(DeepFM, self).__init__(**params)

    def _forward(self, features, is_training):
        embed_column = self.columns_dict.get('embed_fc')
        if not embed_column:
            raise Exception('Need embed_fc given')
        ##### deep part
        deep_net = tf.feature_column.input_layer(features=features, feature_columns=embed_column)
        for layer_id, units in enumerate(self.deep_hidden_units):
            with variable_scope.variable_scope('hiddenlayer_%d' % layer_id) as hidden_layer_scope:
                deep_net = tf.layers.dense(deep_net, units=units, activation=None,
                                           kernel_initializer=self.kernel_initializer, name=hidden_layer_scope)

                if self.batch_norm == 'on' and self.batch_norm_layers[layer_id] == 'on':
                    deep_net = tf.layers.batch_normalization(deep_net, training=is_training, name='bn_' + str(layer_id))

                if self.activation_func is not None:
                    deep_net = self.activation_func(deep_net)

                if not self.dropout and is_training:
                    deep_net = tf.layers.dropout(deep_net, rate=self.dropout)
        ##### fm part
        fm_logits = 0
        feature_size = len(embed_column)
        embed_column.sort(key=lambda x: x.name)
        embed_tensors = [tf.feature_column.input_layer(
            features={embed_column_i.name.split('_')[0]: features[embed_column_i.name.split('_')[0]]},
            feature_columns=embed_column_i) for embed_column_i in embed_column]
        for feature_id_x in range(0, feature_size - 1):
            for feature_id_y in range(feature_id_x, feature_size):
                feature_tensor_x = embed_tensors[feature_id_x]
                feature_tensor_y = embed_tensors[feature_id_y]
                # print(feature_tensor_x.shape)
                # print(feature_tensor_y.shape)
                feature_multiply = tf.multiply(feature_tensor_x, feature_tensor_y)
                fm_logits += tf.reduce_sum(feature_multiply, 1, keepdims=True)
                # print(feature_multiply.shape)
                # print(fm_logits.shape)
        ##### loss part
        deep_logits = tf.layers.dense(deep_net, units=1, activation=None, kernel_initializer=self.kernel_initializer,
                                      name='deep_logits')
        # print(deep_logits.shape)
        all_logits = fm_logits + deep_logits
        # print(all_logits.shape)
        return all_logits
