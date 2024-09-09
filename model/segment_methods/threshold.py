import os
import cv2
import numpy as np
from tqdm import tqdm

class Model():
    def __init__(self, configs):
        self.folder_path = configs.stablizer_out_folder
        self.out_folder = configs.sem_seg_out_folder
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)
        self.thresh = configs.thresh_value
        self.imwrite = configs.write_sem_seg_img

    def object_seg(self, img_path:str, thresh=155, imwrite=False):
        img_gray = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 0)
        _, thresh_img = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
        if imwrite:
            save_name = os.path.split(img_path)[1]
            cv2.imwrite(os.path.join(self.out_folder, save_name), thresh_img)
        return thresh_img

    def __call__(self, img_name_list):
        thresh_imgs_dict = {}
        for _, img_name in enumerate(tqdm(img_name_list)):
            img_path = os.path.join(self.folder_path, img_name)
            if not os.path.exists(img_path):
                continue
            thresh_img = self.object_seg(img_path, self.thresh, self.imwrite)
            thresh_imgs_dict[img_name] = thresh_img
        return thresh_imgs_dict

