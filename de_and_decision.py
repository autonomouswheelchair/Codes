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

def main():

    model_name = "Intel/dpt-swinv2-tiny-256"
    # model_name = "Intel/dpt-hybrid-midas"
    model_dir = "/models"  # Change this to the directory where the volume is mounted

    # Create the models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Check if the model exists in the model directory
    if not os.path.exists(os.path.join(model_dir, model_name)):
        # If not, download the model
        model = DPTForDepthEstimation.from_pretrained(model_name)
        processor = DPTImageProcessor.from_pretrained(model_name)
        # Save the model and processor
        model.save_pretrained(os.path.join(model_dir, model_name))
        processor.save_pretrained(os.path.join(model_dir, model_name))
    else:
        # If it does, load the model and processor from the directory
        model = DPTForDepthEstimation.from_pretrained(os.path.join(model_dir, model_name))
        processor = DPTImageProcessor.from_pretrained(os.path.join(model_dir, model_name))

    # Get the current working directory
    current_directory = os.getcwd()

    # Construct the path to the video file
    video_path = os.path.join(current_directory, "c_b407_video.mp4")

    # Now you can use video_path to access the video file
    print("Video path:", video_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(torch.cuda.is_available())

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output_swinv2_tiny.mp4", fourcc, fps, (width, height))

    current_frame = 0
    start = time.time()
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if current_frame % 15 == 0:
            # Convert frame to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Prepare image for the model
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                # Get depth prediction
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )

            # Move prediction tensor to CPU for visualization
            prediction = prediction.cpu()

            # Visualize the prediction
            output = prediction.squeeze().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            depth = cv2.cvtColor(np.array(Image.fromarray(formatted)), cv2.COLOR_GRAY2BGR)
            # Find the brightest region
            direction = find_brightest_region(depth)

            # Display direction
            cv2.putText(depth, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show frame
            
            out.write(depth)
            cv2.imshow('Frame', depth)
            cv2.waitKey(1)


            print(f'Processed frame {current_frame} of {frame_count}')
        current_frame += 1
        
        '''if current_frame % 2 == 0:  # Print every 10 frames to reduce overhead
            print(f'Processed frame {current_frame} of {frame_count}')
            break'''
    end = time.time()
    print(f'Execution time: {end - start} s')
    print(f'Speed: {current_frame / (end - start)} FPS')

    # Release the video file and writer object, and close windows
    cap.release()
    out.release()
    # cv2.destroyAllWindows() does not work on windoes

if __name__ == "__main__":
    main()
