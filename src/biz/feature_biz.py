from typing import *

import tensorflow as tf
from tensorflow.python.feature_column.feature_column_v2 import FeatureColumn

from bean.feature import Feature
from common import context
from common.constant import *
from common.constant import FieldType
from utils import log_util, safe_util
from utils.args_util import FLAGS
from utils.stream_util import Stream

__all__ = ['parse_feature']


def parse_feature(_feature_info_path: str, _feature_dict_path: str) -> List[Feature]:
    """
    解析特征
    :param _feature_info_path: 特征文件
    :param _feature_dict_path: 特征值文件
    :return: 特征列表
    """
    with open(_feature_info_path) as stream:
        stream: Iterable[str] = stream
        skipFeatureNameList: List[str] = Stream(stream).find_first().strip().split(STR_SKIP_FEATURES)[1].split(
            DELIMITER_COMMA)
        print("APP skipFeatureNameList:", skipFeatureNameList)

        def line_to_feature(line: str) -> Feature:
            line_list: List[str] = line.strip().split(DELIMITER_TAB)
            feature = Feature()
            feature.set_index(int(line_list[0]))
            feature.set_name(line_list[1])
            feature.set_type(line_list[2])
            return feature

        featureList: List[Feature] = Stream(stream).map(line_to_feature) \
            .filter(lambda feature: feature.get_name() not in skipFeatureNameList).to_list()
    with open(_feature_dict_path) as stream:
        stream: Iterable[str] = stream

        def line_to_nameValue(line: str) -> Tuple[str, str]:
            lineList = line.strip().split(DELIMITER_TAB)
            nameValueList = lineList[0].split('-')[1].split('^')
            name: str = nameValueList[0]
            value: str = nameValueList[1]
            return name, value

        nameValueMap: Dict[str, List[Tuple[str, str]]] = Stream(stream).map(line_to_nameValue).group_by(
            lambda nameValue: nameValue[0])
        nameValuesMap: Dict[str, List[str]] = {k: Stream(v).map(lambda x: x[1]).distinct().to_list() for k, v in
                                               nameValueMap.items()}
    Stream(featureList).for_each(
        lambda feature: feature.get_valueList().extend(safe_util.ofList(nameValuesMap.get(feature.get_name()))))
    return featureList


def build_column(featureList: List[Feature], embedding_size, combiner=None, max_norm=None) -> Dict[
    str, List[FeatureColumn]]:
    """

    :param featureList: 特征列表
    :param embedding_size: 离散特征的embedding维数
    :param combiner: combiner
    :param max_norm: max_norm
    :return: List[FeatureColumn]
    """

    def feature_process(feature: Feature, column_result) -> None:
        featureType = feature.get_type()
        featureName = feature.get_name()
        featureValueList = feature.get_valueList()
        if FieldType.STRING == featureType:
            if feature.get_valueList():
                fc = tf.feature_column.categorical_column_with_vocabulary_list(featureName, featureValueList)
                fc_emb = tf.feature_column.embedding_column(fc, dimension=embedding_size,
                                                            initializer=tf.glorot_uniform_initializer)
                fc_indi = tf.feature_column.indicator_column(fc)
                column_result.setdefault('embed_fc', []).append(fc_emb)
                column_result.setdefault('indi_fc', []).append(fc_indi)
        elif FieldType.INT == featureType:
            fc = tf.feature_column.numeric_column(featureName, shape=1, default_value=0, dtype=tf.int64)
            column_result.setdefault('value_fc', []).append(fc)
        elif FieldType.FLOAT == featureType:
            fc = tf.feature_column.numeric_column(featureName, shape=1, default_value=0.0, dtype=tf.float32)
            column_result.setdefault('value_fc', []).append(fc)
        else:
            log_util.err_log('Warning! Wrong feature type {}!'.format(featureType))

    column_result = dict()
    Stream(featureList).for_each(lambda feature: feature_process(feature, column_result))
    return column_result


if __name__ == '__main__':
    _feature_info_path = FLAGS.feature_info
    _feature_dict_path = FLAGS.dict_path
    feature_list = parse_feature(_feature_info_path, _feature_dict_path)
    Stream(feature_list).print()
    result = build_column(context.featureList, 16)
    Stream(result.items()).print()
