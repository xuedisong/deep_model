import math

import numpy as np

N = 65536
K = 4
norm = 1
drop_rate = 0.000000
logR = -2.053009
negS = 0.025
posS = 1

a1 = [1.46901, -0.32852, -0.44634, -2.32427]
b3 = [0.86606, -0.46728, 0.94158, -1.33929]
allFeatures = [a1, b3]
a1 = np.array(a1)
b3 = np.array(b3)

r = 1 / len(allFeatures)

latentSum = a1 + b3
sumSquare = (latentSum ** 2).sum()
squareSum = (a1 ** 2).sum() + (b3 ** 2).sum()
predictVal = (sumSquare - squareSum) / 2
# 计算logits
logits = predictVal * r + logR
# 过sigmoid
predictValue = 1 / (1 + math.exp(-logits))
# 还原
revisedValue = negS * predictValue / ((negS - posS) * predictValue + posS)

# Deep 排序预测格式：
# instance=Map{feature_name:feature_value}
# {“instances": [{'wk':'2-wk^3','country':'3-country^CHINA'},{'wk':'2-wk^3','country':'3-country^CHINA'}]}
# 返回格式{"predictions": [[0.00732928], [0.00732928]]}
# predictValue=predictProbList[index][0] 返回直接是概率。
