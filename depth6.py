import os
import torch
import cv2
import numpy as np
from PIL import Image

# Load the pre-trained depth estimation model
model_path = "D:/pytorch_model.bin"
net = cv2.dnn.readNet(model_path)

# Get the current working directory
current_directory = os.getcwd()

# Construct the path to the video file
video_path = os.path.join(current_directory, "c_b407_video.mp4")

# Now you can use video_path to access the video file
print("Video path:", video_path)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

current_frame = 0
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 1.0, (width, height), (0, 0, 0), swapRB=False, crop=False)

    # Set the blob as input to the network
    net.setInput(blob)

    # Get the depth map from the network
    depth_map = net.forward()[0]

    # Normalize the depth map
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Write the depth-enhanced frame to the output video
    out.write(depth_map)

    current_frame += 1
    if current_frame % 2 == 0:  # Print every 10 frames to reduce overhead
        print(f'Processed frame {current_frame} of {frame_count}')
        break

# Release the video file and writer object, and close windows
cap.release()
out.release()
cv2.destroyAllWindows()