# deep_model

深度模型

## 部署

- 服务器: Mac OS
- python版本: 3.5.2 在此环境下安装不成功tensorflow，所以用3.7

```
conda create --name python37 python=3.7
conda activate python37
conda deactivate
```

- tensorflow版本: 1.13.1
- 1.8版本以后 需要手动开启eager模式
  自测输入函数代码
  输入函数 解析文件，是python原生代码的，没有利用tf的API,可以直接debug.
  但是这些debug出来的可以看出是tensor了，但是tensor里的具体内容是看不出来的。

自测 loss

```python
import tensorflow as tf

tf.__version__
tf.enable_eager_execution()
tf.executing_eagerly()
logit = tf.convert_to_tensor([[1], [2], [3], [4], [5], [6]], dtype=tf.float32)
label = tf.convert_to_tensor([[0], [1], [0], [0], [0], [1]], dtype=tf.float32)
prob = tf.sigmoid(logit)
sample_loss = tf.multiply(50 * label, -tf.log(prob)) + tf.multiply(1 - label, -tf.log(1 - prob))
loss = tf.reduce_sum(sample_loss, 0) / len(sample_loss)
sample_loss = tf.nn.weighted_cross_entropy_with_logits(label, logit, 50)
loss = tf.losses.compute_weighted_loss(sample_loss)
```

自测 数据流

```python
import tensorflow as tf

tf.__version__
tf.enable_eager_execution()
tf.executing_eagerly()
data = tf.range(0, 10)
data = tf.data.Dataset.from_tensor_slices(data)
data1 = data.repeat(None)
for i in data1:
    print(i.numpy())
```

自测 esmm

```python
import tensorflow as tf

tf.__version__
tf.enable_eager_execution()
tf.executing_eagerly()
features = {'wk': [['2-wk^6'],
                   ['2-wk^0'],
                   ['2-wk^1'],
                   ['2-wk^2'],
                   ['2-wk^6']],
            'hr': [['3-hr^08'], ['3-hr^09'], ['3-hr^16'], ['3-hr^23'], ['3-hr^09']]}
fc_wk = tf.feature_column.categorical_column_with_vocabulary_list("wk", vocabulary_list=['2-wk^0', '2-wk^1', '2-wk^2',
                                                                                         '2-wk^6'])
fc_hr = tf.feature_column.categorical_column_with_vocabulary_list("hr",
                                                                  vocabulary_list=['3-hr^08', '3-hr^09', '3-hr^16',
                                                                                   '3-hr^23'])

fc_wk = tf.feature_column.embedding_column(fc_wk, dimension=2)
fc_hr = tf.feature_column.embedding_column(fc_hr, dimension=2)
# fc_hr = tf.feature_column.indicator_column(fc_hr)
fc_wk.name

emb_tensors = [tf.feature_column.input_layer(features=features, feature_columns=fc_wk),
               tf.feature_column.input_layer(features=features, feature_columns=fc_wk)]

deep_net = tf.concat(emb_tensors, 1)
```

自测 embedding table：一定内外分隔符的样本，多值特征会按照最多值的个数，填充任意值 ''，这种值和低频值，或者新值，三种可能都在字典中找不到，都是新的桶（embedding table粒度在一个特征域下），对应着字典数量+1的index，所以embedding table 字典数量+1的emb做成其他emb的均，这样就是刚才三类特征值的embedding了。\
冷启动的embedding参与了反向传播和前向推理。\
多值特征在制作时，当sql的prefix('1-field^feature',array())的array是空时，这个特征值会做出'1-field^null'等，这个当作正常特征字典使用也可以。
多值特征的embedding是多个值的embedding的平均。
```python
import tensorflow as tf

tf.__version__
tf.enable_eager_execution()
tf.executing_eagerly()

# dataset
# feature conf
features_name_list = ['wk', 'hr', 'usermodel']
features = {'wk': [['2-wk^6'],
                   ['2-wk^0'],
                   ['2-wk^1'],
                   ['2-wk^2'],
                   ['2-wk^6']],
            'hr': [['3-hr^08'], ['3-hr^09'], ['3-hr^16'], ['3-hr^23'], ['3-hr^09']]}
daily_feature = '15-usermodel^2023,15-usermodel^1234,15-usermodel^4567,15-usermodel^1101'
daily_feature2 = '15-usermodel^2023,15-usermodel^1234'
n = 2
FIELD_OUTER_DELIM = ' '
FIELD_INNER_DELIM = '\031'
batch_size = len(features['wk'])
tmp1 = [[FIELD_INNER_DELIM.join(daily_feature.split(','))]] * (batch_size - n)
tmp2 = [[FIELD_INNER_DELIM.join(daily_feature2.split(','))]] * n
features.setdefault('usermodel', tmp2 + tmp1)
dataset = []
for i in range(batch_size):
    dataset_line_list = []
    for feature_name_i in features_name_list:
        dataset_line_list.append(features[feature_name_i][i][0])
    dataset.append(FIELD_OUTER_DELIM.join(dataset_line_list))
print(dataset)

# parse dataset to column tensor
MULTI_VALUE_FEATURE = {'usermodel': 5}
value = dataset
feature_num = len(features_name_list)
str_columns = tf.decode_csv(value, [''] * feature_num, field_delim=FIELD_OUTER_DELIM)
for idx, column in enumerate(str_columns):
    sparse_col = tf.strings.split(column, FIELD_INNER_DELIM)
    
    if features_name_list[idx] in MULTI_VALUE_FEATURE:
        dense_shape = tf.concat(
                            [tf.gather_nd(sparse_col.dense_shape, [[0]]), [MULTI_VALUE_FEATURE[features_name_list[idx]]]], 0)
    else:
        dense_shape = sparse_col.dense_shape
    # dense_shape=[sparse_col.dense_shape.numpy()[0],5]
    # dense_col = tf.sparse.to_dense(sparse_col, '')
    dense_col = tf.sparse_to_dense(sparse_col.indices, dense_shape, sparse_col.values, "")    
    features[features_name_list[idx]] = dense_col

# cold start conf 
COLDSTART_NAMES = ['usermodel']
feature_name = 'usermodel'
vocabulary_list = ['15-usermodel^2023', '15-usermodel^1234', '15-usermodel^4567']

embedding_size = 10
embed_matrix = tf.get_variable(name=feature_name + '_embmatrix', shape=[len(vocabulary_list), embedding_size],
                               initializer=tf.initializers.random_normal(mean=0, stddev=0.1), trainable=True)
oov_embed = tf.reduce_mean(embed_matrix, 0) if feature_name in COLDSTART_NAMES else tf.zeros([embedding_size])
embed_all_matrix = tf.concat([embed_matrix, [tf.convert_to_tensor(oov_embed)]], 0, name=feature_name + '_concat')
## test stop gradient not success
# with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
#     oov_embed_var = tf.Variable([0.0] * embedding_size, trainable=False, name='oov_embed_var')
#     oov_embed_var.assign(oov_embed)
#     oov_embed_var_1 = tf.get_variable(name='oov_embed_var',shape=[embedding_size])

DEFAULT_VALUE = -1
table = tf.contrib.lookup.index_table_from_tensor(mapping=vocabulary_list, default_value=DEFAULT_VALUE,
                                                  num_oov_buckets=1)
tags = table.lookup(features[feature_name])
in_vocabulary_idx = tf.where(tf.not_equal(tags, DEFAULT_VALUE))
gather_result = tf.gather_nd(tags, in_vocabulary_idx)
sparse_tags = tf.SparseTensor(in_vocabulary_idx, gather_result, tf.shape(tags, out_type=tf.int64))

embed_lookup = tf.nn.embedding_lookup_sparse(params=embed_all_matrix, sp_ids=sparse_tags, sp_weights=None,
                                             combiner="mean")
print("feature values:{},embedding is:{}".format(features[feature_name], embed_lookup))
```

自测 输入函数的具体内容 见 src/biz/input_biz.py

自测nfm代码

```python
import tensorflow as tf

tf.__version__
tf.enable_eager_execution()
tf.executing_eagerly()

features = {'wk': [['2-wk^6'],
                   ['2-wk^0'],
                   ['2-wk^1'],
                   ['2-wk^2'],
                   ['2-wk^6']],
            'hr': [['3-hr^08'], ['3-hr^09'], ['3-hr^16'], ['3-hr^23'], ['3-hr^09']]}
fc_wk = tf.feature_column.categorical_column_with_vocabulary_list("wk", vocabulary_list=['2-wk^0', '2-wk^1', '2-wk^2',
                                                                                         '2-wk^6'])
fc_hr = tf.feature_column.categorical_column_with_vocabulary_list("hr",
                                                                  vocabulary_list=['3-hr^08', '3-hr^09', '3-hr^16',
                                                                                   '3-hr^23'])

fc_wk = tf.feature_column.embedding_column(fc_wk, dimension=2)
fc_hr = tf.feature_column.embedding_column(fc_hr, dimension=2)
deep_net_concat = tf.feature_column.input_layer(features=features,
                                                feature_columns=[fc_hr,
                                                                 fc_wk])  # 会按照feature_columns=[fc_hr, fc_wk]的顺序拼接tensor
deep_net_hr = tf.feature_column.input_layer(features=features, feature_columns=[fc_hr])
deep_net_wk = tf.feature_column.input_layer(features=features, feature_columns=[fc_wk])
bi_layer = tf.multiply(deep_net_hr, deep_net_wk)
bi_layer = tf.add(bi_layer, tf.multiply(deep_net_hr, deep_net_wk))  # 交叉项 sum pooling,nfm侧
interaction_concat = tf.concat([tf.multiply(deep_net_hr, deep_net_wk),
                                tf.multiply(deep_net_hr, deep_net_wk)], 1)  # 交叉项concat pooling,实验侧
emb_concat = tf.concat([deep_net_hr, deep_net_wk], 1)  # emb concat po0ling,deepfm deep侧
hidden_layer_1 = tf.layers.dense(bi_layer, units=3, activation=tf.nn.relu,
                                 kernel_initializer='he_normal', name='hidden_layer_1')
hidden_layer_2 = tf.layers.dense(hidden_layer_1, units=2, activation=tf.nn.relu,
                                 kernel_initializer='he_normal', name='hidden_layer_2')
logits = tf.layers.dense(hidden_layer_2, units=1, activation=None,
                         kernel_initializer='he_normal', name='logits')  # 思考都是relu时，如果隐层单元少时，logit值很多是0。
all_logits = logits + logits
import operator
import functools

functools.reduce(operator.add, [logits, logits])
```

自测attention代码

```python
import tensorflow as tf

tf.__version__
tf.enable_eager_execution()
tf.executing_eagerly()
a = tf.constant([[1, 1], [2, 2], [3, 3]], dtype=tf.float32)
tf.math.l2_normalize(a, [0, 1])

features = {'user_item': [[0, 1, 2], [1, 2, 2], [0, 2, 3], [1, 2, 3]]}
user_item = tf.contrib.feature_column.sequence_categorical_column_with_vocabulary_list("user_item",
                                                                                       vocabulary_list=[0, 1, 2, 3],
                                                                                       default_value=0)
fc = tf.feature_column.embedding_column(user_item, dimension=2)
input_layer, sequence_length = tf.contrib.feature_column.sequence_input_layer(features, fc)

clicked_model_feature = input_layer  # (4,3,2)
clicked_ptitle_feature = input_layer
clicked_count_features = input_layer
concat_feature = tf.concat([clicked_model_feature, clicked_ptitle_feature, clicked_count_features], axis=-1)  # (4,3,6)
dense_net = tf.layers.dense(inputs=concat_feature,
                            units=clicked_model_feature.shape[2],
                            kernel_initializer=tf.initializers.random_normal(),
                            activation=tf.nn.leaky_relu)  # (4,3,2)
attention_score = tf.layers.dense(inputs=dense_net,
                                  units=1,
                                  kernel_initializer=tf.initializers.random_normal(),
                                  activation=tf.nn.leaky_relu)  # (4,3,1)
attention_score = tf.transpose(attention_score, perm=[0, 2, 1])  # (4,1,3)
attention_score = tf.nn.softmax(attention_score, axis=-1)  # (4,1,3)
attention_output = tf.matmul(attention_score, dense_net)  # (4,1,2)
attention_output = tf.reshape(attention_output, [-1, attention_output.shape[2]])  # (4,2)
attention_output  # B=w2'(Aw1)'(Aw1);A=(n,d1),w1=(d1,k),w2=(k,1),B=(1,k);k=2,n=3 一种pooling的方式。含义就是将某列作为权重值，进一步放大其影响。
# d1=6,从A=(n,d1)变成B=(1,k),C=Aw1=(n,k),B=w2'C'C=(Cw2)'C,Cw2=(n,1),Cw2=softmax(Cw2)

labels = tf.constant([[1, 0], [0, 1], [1, 0]], dtype=tf.float32)
logits = tf.constant([[1, 1], [2, 2], [3, 3]], dtype=tf.float32)
loss = tf.losses.softmax_cross_entropy(labels, logits, scope="loss")
```

双塔方案：loss采用softmax，但是物品向量侧需要由物品属性DNN得来，物品ID使用预训练的物品ID/或者就是物品ID。
这样的思路是，尽量少训练参数，防止模型过拟合，因为大量的one-hot embedding将会产生大量的训练边。所以GCN用户侧到时候也可以直接使用训练好的embedding。

学习tf
```python
import tensorflow as tf

tf.__version__
tf.enable_eager_execution()
tf.executing_eagerly()

x = tf.Variable(2.0)
with tf.GradientTape() as g:
    y = x ** 2
print(g.gradient(y, x))
```
## 注意事项

```
set -e 对 . file.sh 有效
参数虽然是希望可调的，但是在工程逻辑内，是不会产生不同实例的模型的。除非循环训练等等。暂时不考虑。
```

## 代码结构说明

tf 的API，主要是Estimator API，

1. FeatureColumn 构建model网络input layer，可能有点属于特征工程部分。
2. input_fn 中会用到 feature_name,type来构建TFRecord,或者是input tensor
3. 实现**estimator.base_model.BaseModel._forward**

```
todo: 实际上就一个接口：_forward,转换操作，然后就完事了。
因为接口和抽象类，本质区别是自定义成员属性。而成员属性的真正含义，应该是实例在方法执行时，自身属性也发生变化，
但其实这个不需要成员属性。也就是一个biz的函数而已。之前的抽象类，是路径实例在成员属性上，也就是context会发生变化。
所以这个我再改造下。
回应：关于抽象类，抽象类的级别其实是由客观决定的，如果客观上，外界API需要用户提供model_fn，那么就应该是这个级别之下，不能再向上包含了。
所以假设对于某个抽象类级别而言，可以选择抽象级别尽可能地下移，但伴随的是对于被移开的级别，将无法逻辑变动了。所以目前此级别和灵活性等，做到了
1. 跟随客观，低于客观级别model_fn
2. 尽可能保证灵活性，不再下移抽象类级别。使得拥有灵活性。

todo:
应该有 TFRecord 直接得到Tensor DF的一个Tensor 直接输出的形式，即直接衔接起input_fn和model_fn。直接输出。的正向逻辑。
```