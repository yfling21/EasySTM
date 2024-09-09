import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from utils import contour_selection, contours_split, contours2bboxes

class Model():
    def __init__(self, configs):
        self.folder_path = configs.folder_path
        self.out_folder = configs.ins_seg_out_folder
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)
        self.imwrite = configs.write_ins_seg_img
        self.return_bboxes = configs.return_bboxes
        self.tensor_bboxes = configs.tensor_bboxes

    def instance_seg(self, thresh_img, img_name, split=False, imwrite=False):
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contour_selection(contours)
        if split:
            contours = contours_split(contours)
            contours = contour_selection(contours)
        if imwrite:
            bi_seg_img = np.zeros(thresh_img.shape)
            for i, _ in enumerate(contours):
                cv2.drawContours(bi_seg_img, contours, i, (255, 255, 255), -1)
            cv2.imwrite(os.path.join(self.out_folder, img_name), bi_seg_img)
        return contours

    def __call__(self, thresh_imgs_dict):
        processed_num = 0
        instance_dict = {}
        for img_name, thresh_img in tqdm(thresh_imgs_dict.items()):
            split = True if processed_num <= 9 else False
            contours = self.instance_seg(thresh_img, img_name, split, self.imwrite)
            if self.return_bboxes:
                instance_dict[img_name] = contours2bboxes(contours, self.tensor_bboxes)
            else:
                instance_dict[img_name] = contours

        return instance_dict

