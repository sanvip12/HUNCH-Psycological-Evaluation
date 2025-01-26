import os
from time import sleep
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logs
os.environ["OMP_NUM_THREADS"] = "2"      # Limit OpenMP threads
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import tensorflow as tf
import transformers
print(tf.__version__)  # Ensure TensorFlow is installed
print(transformers.__version__)  # Check transformers version
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

import os
import subprocess
from PIL import Image
from transformers import pipeline
import cv2
import numpy as np
import time  # Added for better time tracking
import matplotlib.pyplot as plt

from deepface import DeepFace
from mtcnn import MTCNN
import cv2

# Initialize face detector
detector = MTCNN()

# Initialize webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Detect faces
    faces = detector.detect_faces(frame)

    if faces:
        for face in faces:
            x, y, w, h = face['box']
            cropped_face = frame[y:y + h, x:x + w]

            try:
                # Perform emotion analysis on cropped face
                predictions = DeepFace.analyze(cropped_face, actions=['emotion'], enforce_detection=False)
                dominant_emotion = predictions[0]['dominant_emotion']

                # Draw rectangle and label with dominant emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            except ValueError as e:
                print(f"Could not analyze face: {e}")
                continue

    # Display the frame with detected faces and emotion labels
    cv2.imshow('Emotion Analysis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Initialize emotion counting dictionary
combined_results = {
    'Happy': 0,
    'Sad': 0,
    'Angry': 0,
    'Surprise': 0,
    'Neutral': 0,
    'Fear': 0,
    'Disgust': 0,
}

# Set the path for ffmpeg executable
ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"  # Adjust if ffmpeg is in a different location

# Function to extract frames from a specific time range using ffmpeg
def extract_frames_ffmpeg(video_path, output_dir, start_time, end_time, num_frames):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    duration = end_time - start_time
    if duration <= 0:
        raise ValueError("End time must be greater than start time.")
    
    frame_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    command = [
        ffmpeg_path, '-i', video_path,
        '-vf', f"fps={num_frames/duration}", '-ss', str(start_time), '-to', str(end_time),
        frame_pattern, '-y'
    ]
    
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    
    return [os.path.join(output_dir, f"frame_{i:04d}.jpg") for i in range(1, num_frames)]  # Skip frame_0000.jpg

# Function to get video duration using ffmpeg
def get_video_duration(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    command = [ffmpeg_path, '-i', video_path]
    print(f"Running command to get video duration: {' '.join(command)}")
    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    
    # Print the full output from FFmpeg for debugging
    print("FFmpeg error output:")
    print(result.stderr)
    
    duration_line = [line for line in result.stderr.splitlines() if 'Duration' in line]
    
    if duration_line:
        # Extract duration string and clean it
        duration_str = duration_line[0].split()[1].strip(',')
        print(f"Extracted duration string: {duration_str}")
        try:
            h, m, s = map(float, duration_str.split(':'))
            return h * 3600 + m * 60 + s
        except ValueError as e:
            print(f"Error parsing duration string: {e}")
            raise ValueError("Could not parse the duration.")
    else:
        raise ValueError("Could not retrieve video duration. FFmpeg output doesn't contain 'Duration'.")

# Function to calculate the number of frames to analyze
def calculate_num_frames(video_path, frame_rate, start_time, end_time):
    video_duration = get_video_duration(video_path)
    duration = end_time - start_time
    if duration <= 0:
        raise ValueError("End time must be greater than start time.")
    
    # Calculate frames based on desired frame rate
    return int(frame_rate * duration)

# Initialize the emotion detection pipeline
pipe = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    device=-1  # Use CPU (device=-1 for CPU, change to 0 for GPU if available)
)

# Function to process a single frame
def process_frame(frame):
    try:
        # Convert numpy array frame to PIL image
        pil_image = Image.fromarray(frame)
        results = pipe(pil_image)
        emotion_label = results[0]['label'].capitalize()
        combined_results[emotion_label] += 1
        print(f"Processed frame, Emotion: {emotion_label}")
    except Exception as e:
        print(f"Error processing frame: {e}")
        
frame_rate = 5.6 

# Function to capture video from webcam
def capture_webcam_video(output_path, duration, webcam_index=0):
    # Initialize the webcam capture
    webcam = cv2.VideoCapture(1) 
    #sleep(5)

    if not webcam.isOpened():
        print(f"Error: Could not open webcam with index {webcam_index}.")
        return False

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))  # Adjust resolution as needed

    print("Recording started...")
    start_time = time.time()

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Write the frame to the video file
        out.write(frame)

        # Stop recording after the defined duration
        if time.time() - start_time > duration:
            break

    webcam.release()
    out.release()

    print(f"Recording completed. Video saved to {output_path}")
    return True

# Function to check if video file exists
def check_video_file_exists(video_path):
    if os.path.exists(video_path):
        print("Video file found.")
        return True
    else:
        print(f"Error: Video file not found at {video_path}")
        return False

# Function to process the video range sequentially
def process_video_range_sequential(video_path, start_time, end_time, num_frames):
    output_dir = "./frames"
    frames = extract_frames_ffmpeg(video_path, output_dir, start_time, end_time, num_frames)

    # Process each frame one by one
    for frame_path in frames:
        frame = cv2.imread(frame_path)  # Read frame as numpy array
        process_frame(frame)

# Main function to capture video and process it
output_video_path = r"C:\Users\hongy\Downloads\output_fixed.mp4"  # Full path to the video file
start_time = 0
end_time = 30
webcam_index = 0  # Change this index to the desired webcam

# Step 1: Capture webcam video and save it to file
if capture_webcam_video(output_video_path, duration=30, webcam_index=webcam_index):
    # Step 2: Check if video file exists
    if check_video_file_exists(output_video_path):
        # Step 3: Calculate the number of frames to analyze based on the frame rate
        try:
            num_frames = calculate_num_frames(output_video_path, frame_rate, start_time, end_time)
        except ValueError as e:
            print(f"Error in calculating number of frames: {e}")
            exit()

        # Step 4: Process the video range sequentially
        process_video_range_sequential(output_video_path, start_time, end_time, num_frames)

        print("Webcam processing complete.")
        print("Video processing complete.")
        print(combined_results)

        # Pie chart for emotions
        labels = list(combined_results.keys())
        sizes = list(combined_results.values())

        # Filter out zero values and corresponding labels
        filtered_labels = [label for label, size in zip(labels, sizes) if size > 0]
        filtered_sizes = [size for size in sizes if size > 0]

        # Create the pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(filtered_sizes, labels=filtered_labels, autopct='%1.1f%%', startangle=140)
        plt.title("Emotion Distribution")

        # Show the plot
        plt.show()

        # Print the zero percentages outside the chart if needed
        zero_results = {label: size for label, size in zip(labels, sizes) if size == 0}
        if zero_results:
            print("Zero Percent Emotions:")
            for label, size in zero_results.items():
                print(f"{label}: {size}%") 