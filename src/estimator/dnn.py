from .base_model import BaseModel
from tensorflow.python.ops import variable_scope
import tensorflow as tf
from tensorflow.python.ops.losses import losses


class Dnn(BaseModel):

    def __init__(self, **params):
        super(Dnn, self).__init__(**params)


    def _forward(self, features, is_training):
        embed_column = self.columns_dict.get('embed_fc')
        if not embed_column:
            raise Exception('Need embed_fc given')
	
        deep_net = tf.feature_column.input_layer(features=features, feature_columns=embed_column)

        # layer structrue
        for layer_id, units in enumerate(self.deep_hidden_units):
            with variable_scope.variable_scope('hiddenlayer_%d' % layer_id) as hidden_layer_scope:

                deep_net = tf.layers.dense(deep_net, units=units, activation=None, 
                    kernel_initializer=self.kernel_initializer, name=hidden_layer_scope)

                if self.batch_norm == 'on' and self.batch_norm_layers[layer_id] == 'on':
                    deep_net = tf.layers.batch_normalization(deep_net, training=is_training, name='bn_'+str(layer_id))

                if self.activation_func is not None:
                    deep_net = self.activation_func(deep_net)

                if self.dropout and is_training:
                    deep_net = tf.layers.dropout(deep_net, rate=self.dropout)

        # return deep logits
        with variable_scope.variable_scope('logits') as logits_scope:
            deep_logits = tf.layers.dense(deep_net, units=1, activation=None, kernel_initializer=self.kernel_initializer, name=logits_scope)
        return deep_logits


    def get_model(self, features, labels, mode, params):

        if not self.deep_hidden_units:
            raise ValueError("Need deep_hidden_units given")

        is_training = False
        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training=True

        logits = self._forward(features, is_training)

        loss = losses.sigmoid_cross_entropy(labels, logits)

        return self.get_estimator_spec(mode, logits, loss, labels)

