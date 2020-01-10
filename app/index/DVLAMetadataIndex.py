import json
import numpy as np

class DVLAMetadataIndex():

    location = dict()
    postal_area = dict()
    age_before_median = """March {month1} - Aug {month2}"""
    age_after_median = """Sept {month3} - Feb {month4}"""
    code_before_median = []
    code_after_median = []

    @staticmethod
    def set_memory_tag_data(file_path):
        tags = json.load(open(file_path, 'r'))
        tags_keys = list(tags.keys())
        tag_values = np.array(list(tags.values()))
        DVLAMetadataIndex.location = dict(zip(tags_keys, tag_values[:, 0]))
        DVLAMetadataIndex.postal_area = dict(zip(tags_keys, tag_values[:, 1]))

    @staticmethod
    def set_age_identifier_data():
        before_median = range(2,30,1)
        after_median = range(51,79,1)
        DVLAMetadataIndex.code_before_median = list(map(lambda x: x.zfill(2), before_median))
        DVLAMetadataIndex.code_after_median = list(map(lambda x: x.zfill(2), after_median))

    @staticmethod
    def parse_memory_tag(dvla_memory_tag):
        first_character = dvla_memory_tag[0]
        second_character = dvla_memory_tag[1]

        return DVLAMetadataIndex.location[first_character], DVLAMetadataIndex.postal_area[second_character]

    @staticmethod
    def parse_age_identifier(age_identifier):
        age_identifier = age_identifier.zfill(2)
        if age_identifier in DVLAMetadataIndex.code_after_median:
            return DVLAMetadataIndex.code_after_median[age_identifier]
        elif age_identifier in DVLAMetadataIndex.code_before_median:
            return DVLAMetadataIndex.code_before_median[age_identifier]
