from typing import Tuple

from bean.feature import Feature, FeatureBO
from common.constant import *
from utils.args_util import FLAGS
from utils.stream_util import *
from utils.stream_util import Stream


def featureList2featureBO(_featureList: List[Feature]) -> FeatureBO:
    """
    featureList to FeatureBO
    :param _featureList: featureList
    :return: FeatureBO
    """
    pass


def parse_feature(_feature_info_path: str, _feature_dict_path: str) -> List[Feature]:
    """
    解析特征
    :param _feature_info_path: 特征文件
    :param _feature_dict_path: 特征值文件
    :return: 特征列表
    """
    with open(_feature_info_path) as stream:
        stream: Iterable[str] = stream
        skip_features: List[str] = Stream(stream).find_first().strip().split(STR_SKIP_FEATURES)[1].split(
            DELIMITER_COMMA)
        print("APP skip_features:", skip_features)

        def line_to_feature(line: str) -> Feature:
            line_list: List[str] = line.strip().split(DELIMITER_TAB)
            feature = Feature()
            feature.set_index(int(line_list[0]))
            feature.set_name(line_list[1])
            feature.set_type(line_list[2])
            return feature

        featureList: List[Feature] = Stream(stream).map(line_to_feature).to_list()
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
        nameValueMap = {k: Stream(v).map(lambda x: x[1]).distinct().to_list() for k, v in nameValueMap.items()}
    Stream(featureList).for_each(lambda feature: feature.set_valueList(nameValueMap.get(feature.get_name())))
    return featureList


if __name__ == '__main__':
    _feature_info_path = FLAGS.feature_info
    _feature_dict_path = FLAGS.dict_path
    feature_list = parse_feature(_feature_info_path, _feature_dict_path)
    Stream(feature_list).print()
