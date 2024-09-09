import pickle
import torch
import torch.nn as nn
from model.segment_methods import threshold, post_processor

class Segmentor():
    def __init__(self, configs):
        super(Segmentor, self).__init__()
        self.configs = configs
        self.model_dict = {
            'threshold': threshold,
        }
        self.model = self.model_dict[configs.segment_method].Model(configs)
        self.post_processor = post_processor.Model(configs)

    def __call__(self, img_name_list):
        print('segmenting')
        thresh_imgs_dict = self.model(img_name_list)
        contours_dict = self.post_processor(thresh_imgs_dict)
        print('finish segmentation')

        return contours_dict
