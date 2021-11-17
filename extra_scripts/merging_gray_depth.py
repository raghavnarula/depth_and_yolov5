import os
import shutil
from pathlib import Path
import glob

src_dir1 = '/home/mononoke/GitUploads/depth_and_yolov5/final_results/depth_00/'
src_dir2 = '/home/mononoke/GitUploads/depth_and_yolov5/final_results/kitti_00/'
dst_dir = '/home/mononoke/GitUploads/depth_and_yolov5/final_results/merged_00/'
for i in os.listdir(src_dir1):
    src_addr = src_dir1+i
    dst_addr = dst_dir+i
    shutil.copy(src_addr,dst_addr)

for i in os.listdir(src_dir2):
    src_addr = src_dir2+i
    dst_addr = dst_dir+i
    shutil.copy(src_addr,dst_addr)
