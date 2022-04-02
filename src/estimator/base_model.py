from abc import abstractmethod
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.estimator.export import export_output
from tensorflow.python.summary import summary

# serving
_DEFAULT_SERVING_KEY = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
_CLASSIFY_SERVING_KEY = 'classification'
_REGRESS_SERVING_KEY = 'regression'
_PREDICT_SERVING_KEY = 'predict'


class BaseModel(object):

    def __init__(self, **params):
        self.deep_hidden_units = self._read_param('deep_hidden_units', **params)
        self.deep_optimizer = self._read_param('deep_optimizer', **params)
        self.wide_optimizer = self._read_param('wide_optimizer', **params)
        self.columns_dict = self._read_param('columns_dict', **params)
        self.dropout = self._read_param('dropout', **params)
        self.batch_norm = self._read_param('batch_norm', **params)
        self.batch_norm_layers = self._read_param('batch_norm_layers', **params)
        self.activation_func = self._read_param('activation_func', **params)
        self.kernel_initializer = self._read_param('kernel_initializer', **params)
        self.mp = self._read_param('mp', **params)
        self.embedding_size = self._read_param('embedding_size', **params)


    # read param
    def _read_param(self, name, **params):
        if name not in params:
            raise KeyError('Need necessary param: {}'.format(name))
        return params[name]


    def get_estimator_spec(self, mode, logits, loss, labels):
        logistic = math_ops.sigmoid(logits, name='logistic')
        predictions = {
            'logits' :logits,
            'logistic' :logistic
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            predict_output = export_output.PredictOutput(logistic)
            return tf.estimator.EstimatorSpec(
                mode = mode,
                predictions = predictions,
                export_outputs = {
#                    _DEFAULT_SERVING_KEY: predict_output, 
                    "predict": predict_output
                })

        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode = mode,
                predictions = predictions,
                loss = loss,
                eval_metric_ops = {"auc" : tf.metrics.auc(labels, logistic)})

        else:
            train_op = self.deep_optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(
                mode = mode,
                predictions = predictions,
                loss = loss,
                train_op = train_op,
				eval_metric_ops = {"auc" : tf.metrics.auc(labels, logistic)})


    @abstractmethod
    def _forward(self, feature):
        pass


    @abstractmethod
    def get_model(self, features, labels, mode, params):
        pass


