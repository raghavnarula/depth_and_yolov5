#!/bin/bash
python yolov5/detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source monodepth2/assets/test_image.jpg --project /home/mononoke/Desktop/mixing/final_results/

python monodepth2/depth_prediction_example.py -imgpath monodepth2/assets/test_image.jpg
