import argparse

parser = argparse.ArgumentParser()

# file and dir
parser.add_argument(
    '--feature_info', type=str, required=True, help='feature info file.')
parser.add_argument(
    '--model_dir', type=str, required=True, help='model dir.')
parser.add_argument(
    '--export_dir', type=str, default='', help='which dir to export savedmodel.')
parser.add_argument(
    '--dict_path', type=str, required=True, help='dict file path.')
parser.add_argument(
    '--train_data', type=str, default='', help='data path for model train.')
parser.add_argument(
    '--test_data', type=str, default='', help='data path for model eval.')
parser.add_argument(
    '--output_file', type=str, default='output_pred.txt', help='output pred result in eval mode.')
parser.add_argument(
    '--emb_print_file', type=str, default='emb_print.txt', help='output hook result in emb_print mode.')

# model core
parser.add_argument(
    '--model_name', type=str, default='dnn', help='model to use')
parser.add_argument(
    '--work_mode', type=str, required=True, choices=['train_and_eval', 'eval', 'export', 'emb_print'], help="work mode")

# model train
parser.add_argument(
    '--train_epochs', type=int, default=1, help='Number of training epochs.')
parser.add_argument(
    '--max_steps', type=int, default=100000000, help='Number of training max steps.')
parser.add_argument(
    '--epochs_per_eval', type=int, default=1, help='The number of training epochs to run between evaluations.')
parser.add_argument(
    '--batch_size', type=int, default=10000, help='Number of examples per batch.')
parser.add_argument(
    '--shuffle', type=int, default=1, )

# nn structure
parser.add_argument(
    '--hidden_units', type=str, default='256-128-64-16', help='hidden units of dnn')
parser.add_argument(
    '--embedding_size', type=int, default=32, help='dimension of embedding column')

# nn param
parser.add_argument(
    '--activation_func', type=str, default='relu', help='activation function in neural network')
parser.add_argument(
    '--kernel_initializer', type=str, default='None', help='initializer function in neural network')

# over-fitting
parser.add_argument(
    '--dnn_dropout', type=float, default=0.0, help='dnn dropout')
parser.add_argument(
    '--batch_norm', type=str, default='off', choices=["on", "off"], help='batch normalization')
parser.add_argument(
    '--batch_norm_layers', type=str, default='off-off-off-off', help='batch normalization layers config')

# optimizer
parser.add_argument(
    '--dnn_optimizer', type=str, default='adagrad', choices=['adagrad', 'adam', 'adadelta', 'ftrl'])
parser.add_argument(
    '--lr_optimizer', type=str, default='ftrl', choices=['adagrad', 'adam', 'adadelta', 'ftrl'])
parser.add_argument(
    '--lr_learning_rate', type=float, default=0.01, help='the learning rate of wide part')
parser.add_argument(
    '--dnn_learning_rate', type=float, default=0.01, help='the learning rate of deep part')

# dist env
parser.add_argument("--dist_mode", type=int, default=0,
                    help="distribution mode {0-local, 1-single_dist, 2-multi_dist}")
parser.add_argument("--ps_hosts", type=str, default='localhost:2222',
                    help="Comma-separated list of hostname:port pairs")
parser.add_argument("--worker_hosts", type=str, default='localhost:2223,localhost:2224,localhost:2225',
                    help="Comma-separated list of hostname:port pairs")
parser.add_argument("--job_name", type=str, default='', help="One of 'ps', 'worker'")
parser.add_argument("--task_index", type=int, default=0, help="Index of task within the job")
parser.add_argument("--num_threads", type=int, default=16, help="Number of threads")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
parser.add_argument("--log_steps", type=int, default=1000, help="save summary every steps")
parser.add_argument("--save_checkpoints_secs", type=int, default=10, help="save checkpoints every seconds")
parser.add_argument("--learning_rate", type=float, default=0.0005, help="learning rate")
parser.add_argument("--l2_reg", type=float, default=0.01, help="L2 regularization")
parser.add_argument("--optimizer", type=str, default='Adam', help="optimizer type {Adam, Adagrad, GD, Momentum}")
parser.add_argument("--deep_layers", type=str, default='20,10,5', help="deep layers")
parser.add_argument("--dcn_layers", type=int, default=3, help="deep layers")
parser.add_argument("--dropout", type=str, default='0.7,0.7,0.5', help="dropout rate")
parser.add_argument("--data_dir", type=str, default='', help="data dir")
parser.add_argument("--dt_dir", type=str, default='', help="data dt partition")
parser.add_argument("--servable_model_dir", type=str, default='export',
                    help="export servable model for TensorFlow Serving")
parser.add_argument("--task_type", type=str, default='train', help="task type {train, infer, eval, export}")
parser.add_argument("--clear_existing_model", type=bool, default=False, help="clear existing model or not")

FLAGS, _ = parser.parse_known_args()
