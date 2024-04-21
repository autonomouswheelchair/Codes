import os
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
import cv2
import numpy as np
from PIL import Image
import time

# Function to calculate brightness of a frame
def calculate_brightness(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate mean brightness
    brightness = np.mean(gray)
    return brightness

# Function to find the brightest region
def find_brightest_region(frame):
    # Divide the frame into three regions (left, middle, right)
    height, width = frame.shape[:2]
    third_width = width // 3

    # Calculate brightness of each region
    left_brightness = calculate_brightness(frame[:, :third_width])
    middle_brightness = calculate_brightness(frame[:, third_width:2*third_width])
    right_brightness = calculate_brightness(frame[:, 2*third_width:])

    # Find the region with the least brightness
    min_brightness = min(left_brightness, middle_brightness, right_brightness)

    # Determine direction based on the brightest region
    if min_brightness == left_brightness:
        direction = "LEFT"
    elif min_brightness == right_brightness:
        direction = "RIGHT"
    else:
        direction = "STRAIGHT"

    return direction

# Function to process video frames
def process_video(model, processor, video_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    current_frame = 0
    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if current_frame % 5 == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )

            prediction = prediction.cpu()
            output = prediction.squeeze().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            depth = cv2.cvtColor(np.array(Image.fromarray(formatted)), cv2.COLOR_GRAY2BGR)
            direction = find_brightest_region(depth)
            cv2.putText(depth, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(depth)
            cv2.imshow('Frame', depth)
            cv2.waitKey(1)

            print(f'Processed frame {current_frame} of {frame_count}')

        current_frame += 1

    end = time.time()
    print(f'Execution time: {end - start} s')
    print(f'Speed: {current_frame / (end - start)} FPS')

    cap.release()
    out.release()

def main():
    model_name = "Intel/dpt-swinv2-tiny-256"
    model_dir = "/models"
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(os.path.join(model_dir, model_name)):
        model = DPTForDepthEstimation.from_pretrained(model_name)
        processor = DPTImageProcessor.from_pretrained(model_name)
        model.save_pretrained(os.path.join(model_dir, model_name))
        processor.save_pretrained(os.path.join(model_dir, model_name))
    else:
        model = DPTForDepthEstimation.from_pretrained(os.path.join(model_dir, model_name))
        processor = DPTImageProcessor.from_pretrained(os.path.join(model_dir, model_name))

    current_directory = os.getcwd()
    video_path = os.path.join(current_directory, "c_b407_video.mp4")
    output_path = "output_swinv2_tiny.mp4"

    process_video(model, processor, video_path, output_path)

if __name__ == "__main__":
    main()
