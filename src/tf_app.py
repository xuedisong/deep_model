import tensorflow as tf

import init
from biz import input_biz, model_biz
from data_handler import MappingParser
from utils.args_util import FLAGS


def main(argv):
    mp = MappingParser()
    input_fn_train = lambda: input_biz.input_fn(mp, FLAGS.train_data, epoch_num=FLAGS.train_epochs,
                                                batch_size=FLAGS.batch_size, shuffle=FLAGS.shuffle,
                                                drop_remainder=True)
    train_spec = tf.estimator.TrainSpec(input_fn=input_fn_train, max_steps=FLAGS.max_steps)
    input_fn_eval = lambda: input_biz.input_fn(mp, FLAGS.test_data, epoch_num=None, batch_size=1093862,
                                               shuffle=FLAGS.shuffle)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_eval, steps=1, start_delay_secs=10, throttle_secs=10)
    tf.estimator.train_and_evaluate(model_biz.build_estimator(), train_spec, eval_spec)


if __name__ == '__main__':
    print("init.source_root_path:", init.source_root_path)
    print("FLAGS:", FLAGS.__dict__)
    tf.app.run(main=main)
