#!/bin/bash
python yolov5/detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source /home/mononoke/Desktop/testing2/ --project final_results/

python monodepth2/depth_prediction_example.py -imgpath /home/mononoke/Desktop/testing2/
