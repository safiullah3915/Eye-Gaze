import cv2
import pandas as pd
import numpy as np
from gaze_tracking import GazeTracking
from imutils import face_utils
import dlib



# Initialize Gaze Tracking and Dlib's face detector
gaze = GazeTracking()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this model file is in the same directory

# Load the captured image
image_path = "Me.jpg"
frame = cv2.imread(image_path)
if frame is None:
    raise Exception("Failed to load image from path.")

# Enhance the image resolution
def enhance_image(image):
    src = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, src, -0.5, 0)

frame = enhance_image(frame)

# Analyze light intensity
def analyze_lighting(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2]), np.std(hsv[:, :, 2])  # Average brightness and variability

average_brightness, brightness_variation = analyze_lighting(frame)

# Process the image with gaze tracking
gaze.refresh(frame)
frame = gaze.annotated_frame()

# Detect faces in the image
faces = detector(frame, 0)

# Prepare data for CSV
data = {
    "eye_region_details": [],
    "iris_tuples": [],
    "light_intensity": [],
    "ambient_intensity": [],
    "head_pose": [],
    "head_x": [],
    "head_y": [],
    "head_z": []
}

for face in faces:
    shape = predictor(frame, face)
    if shape:
        shape = face_utils.shape_to_np(shape)
        iris_coords = str(shape[36:42])  # Simplified assumption for iris based on eye landmarks
        data['eye_region_details'].append(iris_coords)
        data['iris_tuples'].append(iris_coords)
        rect_bb = face_utils.rect_to_bb(face)
        data['head_pose'].append(rect_bb)
        data['head_x'].append(rect_bb[0])
        data['head_y'].append(rect_bb[1])
        data['head_z'].append(rect_bb[2])
    else:
        data['eye_region_details'].append(None)
        data['iris_tuples'].append(None)
        data['head_pose'].append(None)
        data['head_x'].append(None)
        data['head_y'].append(None)
        data['head_z'].append(None)

    data['light_intensity'].append(average_brightness)
    data['ambient_intensity'].append(brightness_variation)

# Save the features to a CSV file
df = pd.DataFrame(data)
df.to_csv('eye_features.csv', index=False)
print("Features extracted and saved to eye_features.csv.")
