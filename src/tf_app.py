import tensorflow as tf
from tensorflow_estimator.python.estimator.estimator import Estimator
from tensorflow_estimator.python.estimator.training import EvalSpec, TrainSpec

import init
from biz import input_biz, model_biz
from common import context
from utils.args_util import FLAGS


def main(argv):
    input_fn_train = lambda: input_biz.input_fn(context.featureList, FLAGS.train_data, epoch_num=FLAGS.train_epochs,
                                                batch_size=FLAGS.batch_size, shuffle=FLAGS.shuffle,
                                                drop_remainder=True)
    train_spec: TrainSpec = tf.estimator.TrainSpec(input_fn=input_fn_train, max_steps=FLAGS.max_steps)
    input_fn_eval = lambda: input_biz.input_fn(context.featureList, FLAGS.test_data, epoch_num=None, batch_size=1093862,
                                               shuffle=FLAGS.shuffle)
    eval_spec: EvalSpec = tf.estimator.EvalSpec(input_fn=input_fn_eval, steps=1, start_delay_secs=10, throttle_secs=10)
    estimator: Estimator = model_biz.build_estimator()

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    print("init.source_root_path:", init.source_root_path)
    print("FLAGS:", FLAGS.__dict__)
    tf.app.run(main=main)
