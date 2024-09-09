import pickle
import torch
import torch.nn as nn
from model.track_methods import naive
from utils import track_list2mot_result

class Tracker():
    def __init__(self, configs):
        super(Tracker, self).__init__()
        self.configs = configs
        self.model_dict = {
            'naive': naive,
        }
        self.model = self.model_dict[configs.track_method].Model(configs)
        self.return_mot_result = configs.return_mot_result

    def __call__(self, instances_dict):
        print('tracking')
        track_list = self.model(instances_dict)
        # print(track_list)
        
        if self.return_mot_result:
            mot_result = track_list2mot_result(track_list, instances_dict)
            with open('../labelme/mot_pred.pkl', "wb") as f:
                pickle.dump(mot_result, f)
            print('finish tracking')
            return mot_result
        else:
            print('finish tracking')
            return track_list

