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
```python
import tensorflow as tf
tf.__version__
tf.enable_eager_execution()
tf.executing_eagerly()
a=tf.constant([[1,1],[2,2],[3,3]],dtype=tf.float32)
tf.math.l2_normalize(a, [0,1])

features = {'user_item': [[0, 1, 2], [1, 2, 2], [0, 2, 3], [1, 2, 3]]}
user_item=tf.contrib.feature_column.sequence_categorical_column_with_vocabulary_list("user_item",
                                                                               vocabulary_list=[0, 1, 2, 3],
                                                                               default_value=0)
fc=tf.feature_column.embedding_column(user_item, dimension=2)
input_layer,sequence_length=tf.contrib.feature_column.sequence_input_layer(features, fc)

clicked_model_feature=input_layer #(4,3,2)
clicked_ptitle_feature=input_layer 
clicked_count_features=input_layer
concat_feature = tf.concat([clicked_model_feature, clicked_ptitle_feature, clicked_count_features], axis=-1)#(4,3,6)
dense_net = tf.layers.dense(inputs=concat_feature,
                            units=clicked_model_feature.shape[2],
                            kernel_initializer=tf.initializers.random_normal(),
                            activation=tf.nn.leaky_relu) #(4,3,2)
attention_score = tf.layers.dense(inputs=dense_net,
                                  units=1,
                                  kernel_initializer=tf.initializers.random_normal(),
                                  activation=tf.nn.leaky_relu) #(4,3,1)
attention_score = tf.transpose(attention_score, perm=[0, 2, 1]) # (4,1,3)
attention_score = tf.nn.softmax(attention_score, axis=-1) # (4,1,3)
attention_output = tf.matmul(attention_score, dense_net) # (4,1,2)
attention_output = tf.reshape(attention_output, [-1, attention_output.shape[2]]) #(4,2)
attention_output # B=w2'(Aw1)'(Aw1);A=(n,d1),w1=(d1,k),w2=(k,1),B=(1,k);k=2,n=3 一种pooling的方式。含义就是将某列作为权重值，进一步放大其影响。
#d1=6,从A=(n,d1)变成B=(1,k),C=Aw1=(n,k),B=w2'C'C=(Cw2)'C,Cw2=(n,1),Cw2=softmax(Cw2)

labels=tf.constant([[1,0],[0,1],[1,0]],dtype=tf.float32)
logits=tf.constant([[1,1],[2,2],[3,3]],dtype=tf.float32)
loss=tf.losses.softmax_cross_entropy(labels, logits, scope="loss")
```
双塔方案：loss采用softmax，但是物品向量侧需要由物品属性DNN得来，物品ID使用预训练的物品ID/或者就是物品ID。
这样的思路是，尽量少训练参数，防止模型过拟合，因为大量的one-hot embedding将会产生大量的训练边。所以GCN用户侧到时候也可以直接使用训练好的embedding。
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