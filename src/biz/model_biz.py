import os

import biz.feature_biz
from biz.factory import *
from common import context
from estimator import *
from utils.args_util import FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _get_estimator(model_name):
    if not model_name:
        raise ValueError('Need given model name for estimator')

    if model_name == 'dnn':
        return Dnn
    elif model_name == 'wide_deep':
        return WideDeep
    elif model_name == 'deepfm':
        return DeepFM
    elif model_name == 'deepfms':
        return DeepFMS
    elif model_name == 'deepfmscs':
        return DeepFMSCS
    elif model_name == 'deepfmscsloss':
        return DeepFMSCSLOSS
    else:
        raise ValueError('Unrecognized model name {0}'.format(model_name))


def build_estimator() -> tf.estimator.Estimator:
    # model class and config
    model_class = _get_estimator(FLAGS.model_name)
    # tf.device('/gpu:2')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 1},
                                      gpu_options=gpu_options),
        save_checkpoints_steps=100,
        keep_checkpoint_max=2500)

    # feature column
    columns_dict = biz.feature_biz.build_column(context.featureList, FLAGS.embedding_size)

    # optimizer
    linear_optimizer = get_optimizer(FLAGS.lr_optimizer, learning_rate=FLAGS.lr_learning_rate)
    dnn_optimizer = get_optimizer(FLAGS.dnn_optimizer, learning_rate=FLAGS.dnn_learning_rate)

    # nn structure
    hidden_units = [int(x) for x in FLAGS.hidden_units.strip().split("-")]
    batch_norm_layers = FLAGS.batch_norm_layers.strip().split("-")
    assert len(hidden_units) == len(batch_norm_layers)

    # initializer
    kernel_initializer = get_kernel_initializer(FLAGS.kernel_initializer)

    # activation function
    activation_func = get_activation_func(FLAGS.activation_func)

    # build model class
    model = model_class(
        deep_hidden_units=hidden_units,
        deep_optimizer=dnn_optimizer,
        wide_optimizer=linear_optimizer,
        columns_dict=columns_dict,
        dropout=FLAGS.dnn_dropout,
        batch_norm=FLAGS.batch_norm,
        batch_norm_layers=batch_norm_layers,
        kernel_initializer=kernel_initializer,
        activation_func=activation_func,
        config=run_config,
        embedding_size=FLAGS.embedding_size
    )

    return tf.estimator.Estimator(
        model_dir=FLAGS.model_dir,
        model_fn=model.get_model,
        config=run_config
    )
