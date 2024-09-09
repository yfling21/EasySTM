import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import make_palette

class Visualizer():
    def __init__(self, configs):
        self.folder_path = configs.stablizer_out_folder
        self.out_folder = configs.visualize_out_folder
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

    def draw_frame(self, track_list, instances_dict):
        for drop_idx in tqdm(range(len(track_list))):
            each_out_folder = os.path.join(self.out_folder, 'pin_out' + '{:0>2}'.format(str(drop_idx)))
            if os.path.exists(each_out_folder):
                pass
            else:
                os.makedirs(each_out_folder)
            
            track_list_idx = 0
            # drop_key = list(track_list.keys())[drop_idx]
            palette = make_palette(len(track_list[drop_idx])+1)
            palette_idx = 0

            img_adder_mp = np.zeros((621,621,3), dtype=np.uint8)
            img_adder_region = np.zeros((621,621,3), dtype=np.uint8)
            for frame_idx, img_name in enumerate(instances_dict.keys()):
                img_path = os.path.join(self.folder_path, img_name)

                rbg_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)

                contours = instances_dict[img_name]

                contour_idx = track_list[drop_idx][track_list_idx][0]

                contour = contours[contour_idx]

                img_curr_region = np.zeros((621,621,3), dtype=np.uint8)
                cv2.drawContours(img_curr_region, contours, contour_idx, palette[palette_idx], -1)
                mask_curr_region = (img_curr_region[:,:,0]!=0)
                img_adder_region[mask_curr_region] = (0,0,0)
                mask_subtraction = (img_adder_region[:,:,0]!=0)
                rbg_img[mask_subtraction] = (rbg_img[mask_subtraction] // 2)
                img_mix = cv2.addWeighted(rbg_img, 1, img_adder_region, 0.8, 0)
                cv2.drawContours(img_adder_region, contours, contour_idx, palette[palette_idx], -1)

                x, y, w, h = cv2.boundingRect(contour)
                mid_position = (int(x+w/2), int(y+h/2))
                cv2.circle(img_adder_mp, mid_position, 2, palette[len(track_list[drop_idx])-palette_idx], -1)
                mask = (img_adder_mp[:,:,0]!=0)
                img_mix[mask] = (0,0,0)
                img_mix = cv2.addWeighted(img_mix, 1, img_adder_mp, 1, 0)
                palette_idx += 1

                cv2.drawContours(img_mix, contours, contour_idx, (255, 0, 0), 2)
                text_position = (int(x+w/3), int(y+h/3))
                cv2.putText(img_mix, str(drop_idx), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 2)

                cv2.imwrite(os.path.join(each_out_folder, img_name), img_mix)

                if track_list_idx < (len(track_list[drop_idx]) - 2):
                    track_list_idx += 1
                else:
                    break
    
    def make_video(self, track_list):
        frame_rate = 10
        frame_width = 621
        frame_height = 621
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out_dir = os.path.join(self.out_folder, 'trace_videos')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        for drop_idx in tqdm(range(len(track_list))):
            image_folder = os.path.join(self.out_folder, 'pin_out' + '{:0>2}'.format(str(drop_idx)))

            img_name_list = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder)) if f.endswith(('.png', '.jpg'))]

            video_name = 'trace' + '{:0>2}'.format(str(drop_idx)) + '.mp4'
            video_path = os.path.join(out_dir, video_name)
            out = cv2.VideoWriter(video_path, fourcc, frame_rate, (frame_width, frame_height))

            for image_file in img_name_list:
                frame = cv2.imread(image_file)
                out.write(frame)

            out.release()
            cv2.destroyAllWindows()
    
    def __call__(self, track_list, instances_dict):
        print('visualizing')
        self.draw_frame(track_list, instances_dict)
        self.make_video(track_list)
        print('finish visualization')

