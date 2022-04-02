# from typing import List, Mapping
#
# from bean.feature import FeatureBO, Feature
# from utils.args_util import FLAGS
# from utils.stream_util import Stream
#
# feature_info_path = FLAGS.feature_info
# feature_dict_path = FLAGS.dict_path
#
#
# def buildFeatureInfo(_feature_info_path, _feature_dict_path) -> FeatureInfo:
#     """
#     创建特征信息
#     :param _feature_info_path: 特征文件
#     :param _feature_dict_path: 特征值文件
#     :return: 特征信息
#     """
#     pass
#
#
# def parseFeatureList(_feature_info_path: str) -> List[Feature]:
#     """
#     解析特征列表
#     :param _feature_info_path: 特征文件
#     :return: 特征列表
#     """
#     result = dict()
#
#
#     print('Finish feature info parse, feature num is {}'.format(len(self.feature_list)))
#     print(self.feature_list)
#     return result
#
#
# def parseFeatureValuesMap(_feature_dict_path: str) -> Mapping[str, List[str]]:
#     """
#     解析特征取值列表
#     :param _feature_dict_path: 特征值文件
#     :return: 特征值列表
#     """
#     vocabulary_map = dict()
#     with open(dict_info_file) as stream:
#         for line in stream:
#             data = line.strip().split('\t')
#             feature_name = data[0].split('-')[1].split('^')[0]
#             vocabulary_map.setdefault(feature_name, []).append(data[0])
#     print('Finish dict info parse, vocabulary feature number is {}'.format(len(vocabulary_map)))
#     print(vocabulary_map.keys())
#     return vocabulary_map
