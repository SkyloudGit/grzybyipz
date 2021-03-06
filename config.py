import os

class Config(object):
    DEBUG = False
    TESTING = False
    JSON_SORT_KEYS = False
    SECRET_KEY = '\xfd{H\xe5<\x95\xf9\xe3\x96.5\xd1\x01O<!\xd5\xa2\xa0\x9fR"\xa1\xa8'
    # BOX_INFERENCE_GRAPH_PATH = os.getcwd() + '/my_model_2/saved_model'
    # NUMBER_INFERENCE_GRAPH_PATH = os.getcwd() + '/my_model_3/saved_model'
    # NUMBER_LABEL_MAP_PATH = os.getcwd() + '/my_model_3/label_map.pbtxt'


class ProductionConfig(Config):
    pass


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True