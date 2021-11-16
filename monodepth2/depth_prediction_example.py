''' 
from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import networks
from utils import download_model_if_doesnt_exist

from pathlib import Path
import glob
import re
# Path Detector ...
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    print(path,"I am printing the path")
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


# Argument Parser 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-imgpath", "--ImagePath", help = "Image Path")
# parser.add_argument("-savepath",)
args = parser.parse_args()


# Setting up networks and loading weights 

model_name = "mono_640x192"
# download_model_if_doesnt_exist(model_name)
encoder_path = os.path.join("models", model_name, "encoder.pth")
depth_decoder_path = os.path.join("models", model_name, "depth.pth")

# LOADING PRETRAINED MODEL
encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
depth_decoder.load_state_dict(loaded_dict)

encoder.eval()
depth_decoder.eval();

# Loading the test image and preprocessing

# image_path = "/home/mononoke/Desktop/mixing/monodepth2/assets/test_image.jpg"
image_path = args.ImagePath
input_image = pil.open(image_path).convert('RGB')
original_width, original_height = input_image.size

feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

# Prediction using tyhe Pytorch Model
with torch.no_grad():
    features = encoder(input_image_pytorch)
    outputs = depth_decoder(features)

disp = outputs[("disp", 0)]

# Saving the results
disp_resized = torch.nn.functional.interpolate(disp,(original_height, original_width), mode="bilinear", align_corners=False)

# Saving colormapped depth image
disp_resized_np = disp_resized.squeeze().cpu().numpy()
vmax = np.percentile(disp_resized_np, 95)

# save_dir = increment_path(Path("../final_results/") / "exp", exist_ok=False)  # increment run
# (save_dir / 'labels').mkdir(parents=True, exist_ok=True)
save_path = "final_results/exp/depth.png"
plt.imsave(arr=disp_resized_np, cmap='magma', vmax=vmax,fname=save_path)
print("Image saved in final_results/exp as depth.png")

'''
# Modified by <r-narula>

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import networks
from utils import download_model_if_doesnt_exist

from pathlib import Path
import glob
import re
# Path Detector ...


# Argument Parser 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-imgpath", "--ImagePath", help = "Image Path")
parser.add_argument("-savepath","--SavePath",help="Path of The Image To Be Saved",default=Path.cwd())
# Upper line for running from bash script 

# Below line for running as python file
# parser.add_argument("-savepath","--SavePath",help="Path of The Image To Be Saved",default=Path.cwd().parent)

args = parser.parse_args()


# Setting up networks and loading weights 

model_name = "mono_640x192"
download_model_if_doesnt_exist(model_name)
encoder_path = os.path.join("models", model_name, "encoder.pth")
depth_decoder_path = os.path.join("models", model_name, "depth.pth")

# LOADING PRETRAINED MODEL
encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
depth_decoder.load_state_dict(loaded_dict)

encoder.eval()
depth_decoder.eval();

# Loading the test image and preprocessing

image_paths = args.ImagePath
print(Path(image_paths).stem)
if not os.path.isdir(str(args.SavePath)+f"/final_results/depth_{Path(image_paths).stem}/"):
    os.mkdir(str(args.SavePath)+f"/final_results/depth_{Path(image_paths).stem}")
    print(f"Created Folder For storing Images in ../final_results ::: depth_{Path(image_paths).stem}")

'''
try:
    osList = os.listdir(image_paths)
    count = 0
    for image_path in glob.glob(image_paths+"/*",recursive=False):

        input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = input_image.size  

        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

        input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

        # Prediction using tyhe Pytorch Model
        with torch.no_grad():
            features = encoder(input_image_pytorch)
            outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]

        # Saving the results
        disp_resized = torch.nn.functional.interpolate(disp,(original_height, original_width), mode="bilinear", align_corners=False)

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        save_path = str(args.SavePath) + f"/final_results/depth_{Path(image_paths).stem}/depth_{osList[count]}"
        plt.imsave(arr=disp_resized_np, cmap='magma', vmax=vmax,fname=save_path)
        count += 1
        print("Image Saved in final_results",count)


except NotADirectoryError:
    # This is a file and not a folder of images

    input_image = pil.open(image_paths).convert('RGB')
    original_width, original_height = input_image.size  

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

    input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

    # Prediction using tyhe Pytorch Model
    with torch.no_grad():
        features = encoder(input_image_pytorch)
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]

    # Saving the results
    disp_resized = torch.nn.functional.interpolate(disp,(original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)

    # split the string name and get the last one 
    save_path = str(args.SavePath) + f"/final_results/depth/depth_{Path(image_paths).stem}.jpg"
    plt.imsave(arr=disp_resized_np, cmap='magma', vmax=vmax,fname=save_path)
    print(f"Stored as {Path(save_path)}")
'''