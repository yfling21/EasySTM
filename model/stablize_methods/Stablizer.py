import pickle
import torch
import torch.nn as nn
from model.stablize_methods import Identical

class Stablizer():
    def __init__(self, configs):
        super(Stablizer, self).__init__()
        self.configs = configs
        self.model_dict = {
            'Identical': Identical,
        }
        self.model = self.model_dict[configs.stablize_method].Model(configs)

    def __call__(self, img_name_list):
        print('stablizing')
        self.model(img_name_list)
        print('finish stablization')
