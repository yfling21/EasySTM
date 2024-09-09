import os
import cv2
import numpy as np
from tqdm import tqdm


class Model():
    def __init__(self, configs):
        self.folder_path = configs.folder_path
        self.out_folder = configs.stablizer_out_folder
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

    def identical_transform(self, img):
        return img

    def __call__(self, img_name_list):
        for _, img_name in enumerate(tqdm(img_name_list)):
            img_path = os.path.join(self.folder_path, img_name)
            if not os.path.exists(img_path):
                continue
            rbg_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
            transformed_img = self.identical_transform(rbg_img)
            cv2.imwrite(os.path.join(self.out_folder, img_name), transformed_img)

