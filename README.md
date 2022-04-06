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
