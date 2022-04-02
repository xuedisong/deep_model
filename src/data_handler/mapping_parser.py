import sys
from utils.args_util import FLAGS

COLUMN_TITLE = ['index', 'name', 'type']


class MappingParser(object):

    def __init__(self):
        feature_info_file = FLAGS.feature_info
        dict_info_file = FLAGS.dict_path
        self.skip_features = set()
        self.feature_list = []
        self.feature_info = self._parse_feature_info(feature_info_file)
        self.vocabulary_map = self._parse_dict_info(dict_info_file)

        assert len(self.feature_list) == len(self.vocabulary_map)

    def _parse_feature_info(self, feature_info):
        result = dict()

        with open(feature_info) as stream:
            for line in stream:

                tmp = line.strip().split('skip_features:')
                if len(tmp) == 2:
                    self.skip_features = set(tmp[1].split(','))

                if line.startswith('#'):
                    continue

                info = line.strip().split('\t')
                if len(info) < 3:
                    raise ValueError("Wrong format for feature info line: {}".format(line))
                if info[1] in self.skip_features:
                    continue

                result[info[1]] = dict(zip(COLUMN_TITLE, info))
                self.feature_list.append(info[1])
        print('Finish feature info parse, feature num is {}'.format(len(self.feature_list)))
        print(self.feature_list)
        return result

    def _parse_dict_info(self, dict_info_file):
        vocabulary_map = dict()
        with open(dict_info_file) as stream:
            for line in stream:
                data = line.strip().split('\t')
                feature_name = data[0].split('-')[1].split('^')[0]
                vocabulary_map.setdefault(feature_name, []).append(data[0])
        print('Finish dict info parse, vocabulary feature number is {}'.format(len(vocabulary_map)))
        print(vocabulary_map.keys())
        return vocabulary_map


if __name__ == '__main__':
    feature_info_file = '../../data/data_demo/feature_info.txt'
    dict_info_file = '../../data/data_demo/dict.txt'
    mp = MappingParser(feature_info_file, dict_info_file)
    print(mp.feature_info)
    print(mp.feature_list)
# print(mp.vocabulary_map)
