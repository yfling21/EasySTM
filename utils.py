import cv2
import torch
import numpy as np
from tqdm import tqdm

# segmentor
def contour_selection(contours):
    avail_contours = []
    for i, contour in enumerate(contours):
        cnt_area = cv2.contourArea(contour)
        if cnt_area < 250:
            continue
        if cnt_area/len(contour) < 5:
            continue
        check_contour = (contour == [620,620])
        check_contour = check_contour.sum(axis=2)
        if np.any(check_contour == 2):
            continue
        check_contour = (contour == [620,0])
        check_contour = check_contour.sum(axis=2)
        avail_contours.append(i)
    new_contours = []
    for i in range(len(avail_contours)):
        new_contours.append(contours[avail_contours[i]])
    return new_contours

def contour_split(contour):
    max_erode = 12
    contours = []
    contours.append(contour)
    img = np.zeros((621,621), dtype=np.uint8)
    cv2.drawContours(img, contours, 0, (255), -1)
    img02 = img.copy()
    num_split_contour_list = []
    for _ in range(max_erode):
        img = cv2.erode(img, kernel=np.ones((3,3),np.uint8), iterations=1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        num_split_contour_list.append(len(contours))
    proper_erode_iter = num_split_contour_list.index(max(num_split_contour_list)) + 2
    img02 = cv2.erode(img02, kernel=np.ones((3,3),np.uint8), iterations=proper_erode_iter)
    contours, _ = cv2.findContours(img02, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    new_contours = []
    for i in range(len(contours)):
        img03 = np.zeros((621,621), dtype=np.uint8)
        cv2.drawContours(img03, contours, i, (255), -1)
        img03 = cv2.dilate(img03, kernel=np.ones((3,3),np.uint8), iterations=proper_erode_iter)
        contours_tmp, _ = cv2.findContours(img03, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        new_contours.append(contours_tmp[0])
    return new_contours

def contours_split(contours:list):
    extend_contours_list = []
    for i in range(len(contours)-1, -1, -1):
        contour = contours[i]
        cnt_area = cv2.contourArea(contour)
        convexity = cnt_area/len(contour)/len(contour)
        if convexity < 0.06:
            contours.pop(i)
            split_contours = contour_split(contour)
            extend_contours_list.extend(split_contours)
    contours.extend(extend_contours_list)
    return contours

def contours2bboxes(contours, tensor_bboxes:bool):
    if tensor_bboxes:
        bboxes = torch.zeros((len(contours),5))
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            bboxes[i,:] = torch.tensor([x, y, x+w, y+h, 1.0])
        return bboxes
    else:
        bboxes = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append(np.expand_dims(np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]]), axis=1))
        return bboxes

# tracker
def initialize_alive_list_track_list(contours, img_name):
    num_object_track = len(contours)
    print(f'{num_object_track} object to track')

    alive_list = []
    track_list = []
    for i in range(num_object_track):
        alive_list.append(i)
        track_list.append([(i,img_name)])
    return alive_list, track_list

def contours2mask(contours):
    mask = np.zeros((len(contours), 621, 621), dtype=np.uint8)
    for i, _ in enumerate(contours):
        cv2.drawContours(mask[i], contours, i, (255), thickness=cv2.FILLED)
    mask = (mask != 0)
    return mask

def calculate_cam_loc(contours01, contours02):
    mask01 = contours2mask(contours01)
    mask02 = contours2mask(contours02)

    cam_loc = np.zeros((len(mask01), len(mask02)), dtype=np.float16)
    for i in range(len(mask01)):
        for j in range(len(mask02)):
            iou = (mask01[i]*mask02[j]).sum()
            corr = iou/(mask01[i].sum())
            cam_loc[i][j] = corr
    return cam_loc

def find_abs_idx(track_list, alive_list, relative_idx_left):
    abs_idx = []
    for i in range(len(track_list)):
        if i not in alive_list:
            continue
        if track_list[i][-1][0] == relative_idx_left:
            abs_idx.append(i)
    return abs_idx

def array_not_in(array, list):
    for item in list:
        if (array == item).all():  
            return False
    return True

def track_list2mot_result(track_list, instance_dict):
    mot_result = {}
    for frame_idx in range(len(instance_dict)):
        img_name = list(instance_dict.keys())[frame_idx]
        mot_result[img_name] = []
    
    for object_idx in range(len(track_list)):
        for alive_idx in range(len(track_list[object_idx])):
            img_name = track_list[object_idx][alive_idx][1]
            bbox_idx = track_list[object_idx][alive_idx][0]
            bbox = instance_dict[img_name][bbox_idx].squeeze(1)
            ixyp = np.concatenate((np.array([object_idx]), bbox[0], bbox[2], np.array([1])), axis=0)
            mot_result[img_name].append(ixyp)
    
    mot_result_post = {}
    for key, value in tqdm(mot_result.items()):
        unique_bbox = []
        unique_bbox_withidx = []
        for i in range(len(value)):
            if array_not_in(value[i][1:], unique_bbox):
                unique_bbox.append(value[i][1:])
                unique_bbox_withidx.append(value[i])
        mot_result_post[key] = np.stack(unique_bbox_withidx, axis=0)
    return mot_result_post

# visualizer
def make_palette(num):
    palette = []
    for i in np.linspace(0,180,num):
        i = int(i)
        palette.append((255,i,i))
    return palette


