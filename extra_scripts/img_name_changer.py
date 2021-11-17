# Change the file name in 0000000.png to 000000_gray.png in the 
from pathlib import Path
import glob
import os
import shutil

src_dir = "/home/mononoke/Desktop/data_odometry_gray/dataset/sequences/00/image_1/"
dst_dir = "../final_results/kitti_00/"
for i in os.listdir(src_dir):
    print(i)
    src_addr = src_dir + i
    print(src_addr)
    img_number = Path(i).stem
    print(img_number)
    dst_addr = dst_dir + f"{img_number}_gray.png"
    print(dst_addr)
    shutil.copy(src_addr,dst_addr)

