import pickle
import torch
import torch.nn as nn
from model.stablize_methods.Stablizer import Stablizer
from model.segment_methods.Segmentor import Segmentor
from model.track_methods.Tracker import Tracker
from model.visualize.Visualizer import Visualizer

class STMTool():
    def __init__(self, configs):
        super(STMTool, self).__init__()
        self.configs = configs
        self.whether_stablize = configs.whether_stablize
        self.stablizer = Stablizer(configs)
        self.segmentor = Segmentor(configs)
        self.tracker = Tracker(configs)
        self.visualizer = Visualizer(configs)

    def __call__(self, img_name_list):
        if self.whether_stablize:
            self.stablizer(img_name_list)
        instances_dict = self.segmentor(img_name_list)
        track_result = self.tracker(instances_dict)
        self.visualizer(track_result, instances_dict)
        return track_result
        
