import os

import yaml


class DeepVoConf:
    conf = None

    @staticmethod
    def get(cfg=None):
        if cfg is not None:
            DeepVoConf.conf = DeepVoConf(cfg)
        return DeepVoConf.conf

    @staticmethod
    def g():
        return DeepVoConf.get()

    def __init__(self, path):
        if path:
            with open(path, 'r') as fp:
                self.conf = yaml.load(fp)

    def __getitem__(self, key):
        return self.conf[key]


class ConfigManager:
    cfg_common = DeepVoConf(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../confs/common.yaml'))
    cfg_model = None

    @staticmethod
    def common():
        return ConfigManager.cfg_common

    @staticmethod
    def model():
        return ConfigManager.cfg_model

    @staticmethod
    def set_cfg(model):
        ConfigManager.cfg_model = DeepVoConf(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../confs/%s.yaml' % model))
