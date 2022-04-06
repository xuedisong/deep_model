from abc import abstractmethod

import tensorflow as tf
from tensorflow.python.estimator.export import export_output
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import losses

_DEFAULT_SERVING_KEY = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
_CLASSIFY_SERVING_KEY = 'classification'
_REGRESS_SERVING_KEY = 'regression'
_PREDICT_SERVING_KEY = 'predict'


def _read_param(name, **params):
    if name not in params:
        raise KeyError('Need necessary param: {}'.format(name))
    return params[name]


class BaseModel(object):

    def __init__(self, **params):
        self.deep_hidden_units = _read_param('deep_hidden_units', **params)
        self.deep_optimizer = _read_param('deep_optimizer', **params)
        self.wide_optimizer = _read_param('wide_optimizer', **params)
        self.featureList = _read_param('featureList', **params)
        self.columns_dict = _read_param('columns_dict', **params)
        self.dropout = _read_param('dropout', **params)
        self.batch_norm = _read_param('batch_norm', **params)
        self.batch_norm_layers = _read_param('batch_norm_layers', **params)
        self.activation_func = _read_param('activation_func', **params)
        self.kernel_initializer = _read_param('kernel_initializer', **params)
        self.embedding_size = _read_param('embedding_size', **params)

    def get_model(self, features, labels, mode, params):
        if not self.deep_hidden_units:
            raise ValueError("Need deep_hidden_units given")
        is_training = False
        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True
        all_logits = self._forward(features, is_training)
        loss = self._get_loss(labels, all_logits)
        return self._get_estimator_spec(mode, all_logits, loss, labels)

    @abstractmethod
    def _forward(self, feature, is_training):
        pass

    def _get_loss(self, labels, logits):
        return losses.sigmoid_cross_entropy(labels, logits)

    def _get_estimator_spec(self, mode, logits, loss, labels):
        logistic = math_ops.sigmoid(logits, name='logistic')
        predictions = {
            'logits': logits,
            'logistic': logistic
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            predict_output = export_output.PredictOutput(logistic)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    #                    _DEFAULT_SERVING_KEY: predict_output,
                    "predict": predict_output
                })

        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops={"auc": tf.metrics.auc(labels, logistic)})

        else:
            train_op = self.deep_optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                eval_metric_ops={"auc": tf.metrics.auc(labels, logistic)})
