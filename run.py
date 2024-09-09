import argparse
import pandas as pd
from model.STMTool import STMTool
import pdb


def get_configs():
    parser = argparse.ArgumentParser(description='STMTool')

    # dataset
    parser.add_argument('--folder_path', type=str, default=r'', help='')
    
    # stablizer
    parser.add_argument('--whether_stablize', type=bool, default=True, help='')
    parser.add_argument('--stablize_method', type=str, default='Identical',
                        help='options: [Identical, ...]')
    parser.add_argument('--stablizer_out_folder', type=str, default='out/stablizer', help='')

    # segmentor
    parser.add_argument('--segment_method', type=str, default='threshold',
                        help='options: [threshold, ...]')
    parser.add_argument('--write_sem_seg_img', type=bool, default=True, help='')
    parser.add_argument('--sem_seg_out_folder', type=str, default='out/sem_seg', help='')
    parser.add_argument('--thresh_value', type=int, default=155, help='')
    parser.add_argument('--write_ins_seg_img', type=bool, default=True, help='')
    parser.add_argument('--ins_seg_out_folder', type=str, default='out/ins_seg', help='')
    parser.add_argument('--return_bboxes', type=bool, default=False, help='')
    parser.add_argument('--tensor_bboxes', type=bool, default=False, help='')

    # tracker
    parser.add_argument('--track_method', type=str, default='naive',
                        help='options: [naive, ...]')
    parser.add_argument('--return_mot_result', type=bool, default=False, help='')
    
    # visualizer
    parser.add_argument('--visualize_out_folder', type=str, default='out/visualize', help='')
    return parser


if __name__ == '__main__':
    parser = get_configs()
    args = parser.parse_args()
    print(args)

    stmtool = STMTool(args)
    print(stmtool)

    # Requiring a xlsx file with image names in the first column
    df = pd.read_excel('dataset01.xlsx', engine='openpyxl')
    img_name_list = df.iloc[:,0].tolist()

    track_result = stmtool(img_name_list)
