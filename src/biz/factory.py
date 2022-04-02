import tensorflow as tf
from estimator import *
from tensorflow.python.ops import init_ops


def get_estimator(model_name):
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


def get_optimizer(optimizer_name, **kargs):
    learning_rate = kargs['learning_rate']
    if not learning_rate:
        raise ValueError('Need given learning rate for optimizer')

    if optimizer_name == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif optimizer_name == 'adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer_name == 'adadelta':
        return tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    elif optimizer_name == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate=learning_rate)
    else:
        raise ValueError('Unrecognized optimizer name {0}'.format(optimizer_name))


def get_kernel_initializer(initializer_name):
	kernel_initializer = None

	if initializer_name == 'he_uniform':
		kernel_initializer = tf.keras.initializers.he_uniform()
	elif initializer_name == 'he_normal':
		kernel_initializer = tf.keras.initializers.he_normal()
	elif initializer_name == 'glorot_uniform':
		kernel_initializer = init_ops.glorot_uniform_initializer()
	return kernel_initializer


def get_activation_func(name):
	if name == 'relu':
		return tf.nn.relu
	elif name == 'sigmoid':
		return tf.nn.sigmoid
	elif name == 'tanh':
		return tf.nn.tanh
	else:
		raise ValueError("Unrecgnized activation func name " + name)
	# custom activation function


if __name__ == '__main__':
	initializer = get_kernel_initializer('he_uniform')
	print(initializer)
