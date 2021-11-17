import os
import glob 
from natsort import natsorted
from pathlib import Path

print(natsorted(glob.glob('/home/mononoke/GitUploads/depth_and_yolov5/final_results/merged_00/*.png')))

# Now we have the images like 00000_depth.png 00000_gray.png 0000001_depth.png 0000001_gray.png and so onn.....
