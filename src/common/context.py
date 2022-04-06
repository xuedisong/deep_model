from typing import List

from bean.feature import Feature
from biz import feature_biz
from utils.args_util import FLAGS

featureList: List[Feature] = feature_biz.parse_feature(FLAGS.feature_info, FLAGS.dict_path)
