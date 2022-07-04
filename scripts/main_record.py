import cv2
import numpy as np
import pyzed.sl as sl
import sys

# Create a ZED camera object
zed = sl.Camera()

# Enable recording with the filename specified in argument
output_path = sys.argv[0]
err = zed.enable_recording(output_path, sl.SVO_COMPRESSION_MODE.H264)

while !exit_app :
    # Each new frame is added to the SVO file
    zed.grab()

# Disable recording
zed.disable_recording()
