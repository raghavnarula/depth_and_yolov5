# Make a video from the Images in Kitti Dataset..
import os
import cv2
import glob
import natsort
import argparse
import numpy as np
from tqdm import tqdm


# Argument Passing
parser = argparse.ArgumentParser()
parser.add_argument("-imgpath", "--ImagePath", help = "Image Path")
args = parser.parse_args()

# Frame to Video Conversion
img_array = []
# Change the Path 
ll = glob.glob(f"{args.ImagePath}/*.png")


for filename in tqdm(natsort.natsorted(ll)):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('final_results/kitti_img2Video.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


# Sparse Optical Flow 

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# The video feed is read in as a VideoCapture object
cap = cv2.VideoCapture("final_results/kitti_img2Video.mp4")
# Variable for color to draw optical flow track
color = (0, 255, 0)
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
# https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
# Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
mask = np.zeros_like(first_frame)


out = cv2.VideoWriter('final_results/sparseOpticalFlw_video.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 1, (1241,376))

while(cap.isOpened()):
    try:
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculates sparse optical flow by Lucas-Kanade method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
        prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
        next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
        # Selects good feature points for previous position
        good_old = prev[status == 1].astype(int)
        # Selects good feature points for next position
        good_new = next[status == 1].astype(int)
        # Draws the optical flow tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            # Draws line between new and old position with green color and 2 thickness
            mask = cv2.line(mask, (a, b), (c, d), color, 2)
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            frame = cv2.circle(frame, (a, b), 3, color, -1)
        # Overlays the optical flow tracks on the original frame
        output = cv2.add(frame, mask)
        # Updates previous frame
        prev_gray = gray.copy()
        # Updates previous good feature points
        prev = good_new.reshape(-1, 1, 2)
        # Opens a new window and displays the output frame
        # cv2.imshow("sparse optical flow", output)
        out.write(output)
        
        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key

        # Change the speed of the video to observe the changes ...
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    except Exception as e:
        break
# The following frees up resources and closes all windows
cap.release()
out.release()
cv2.destroyAllWindows()
