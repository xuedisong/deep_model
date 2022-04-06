from typing import List

from bean.feature import Feature
from biz.feature_biz import *
from utils.args_util import FLAGS

featureList: List[Feature] = parse_feature(FLAGS.feature_info, FLAGS.dict_path)
