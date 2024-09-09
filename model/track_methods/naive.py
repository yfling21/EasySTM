import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import initialize_alive_list_track_list, calculate_cam_loc, find_abs_idx


class Model():
    def __init__(self, configs):
        pass

    def __call__(self, instances_dict):
        frame_num = len(instances_dict)
        for frame_idx in tqdm(range(frame_num - 1)):
            if frame_idx == 0:
                first_frame_key = list(instances_dict.keys())[0]
                alive_list, track_list = initialize_alive_list_track_list(
                    instances_dict[first_frame_key], first_frame_key)
            img01_name = list(instances_dict.keys())[frame_idx]
            img02_name = list(instances_dict.keys())[frame_idx+1]
            contours01 = instances_dict[img01_name]
            contours02 = instances_dict[img02_name]

            cam_loc = calculate_cam_loc(contours01, contours02)

            track_list_one_iter = []
            for relative_idx_left in range(len(contours01)):
                if cam_loc[relative_idx_left].max() == 0:
                    abs_idx_list = find_abs_idx(track_list, alive_list, relative_idx_left)
                    if abs_idx_list == []:
                        continue
                    for abs_idx in abs_idx_list:
                        alive_list.pop(alive_list.index(abs_idx))
                        print('alive object index:', alive_list)
                relative_idx_right = cam_loc[relative_idx_left].argmax()
                abs_idx_list = find_abs_idx(track_list, alive_list, relative_idx_left)
                if abs_idx_list == []:
                    continue
                for abs_idx in abs_idx_list:
                    track_list_one_iter.append((abs_idx, relative_idx_right))
            
            for tup in track_list_one_iter:
                track_list[tup[0]].append((tup[1], img02_name))

        return track_list

