#!/bin/bash
# python yolov5/detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source testing2/ --project final_results/

python monodepth2/depth_prediction_example.py -imgpath DeepVO-pytorch/KITTI/images/00/

# python monodepth2/depth_prediction_example.py -imgpath testing2/

# python opticalFlow/opticalFlw.py -imgpath testing2/