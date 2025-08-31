import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from gaze_tracking import GazeTracking
from imutils import face_utils
import dlib
import pickle
import time
import threading
import os

# Load the trained model and the scaler
model = load_model('psychological_state_predictor.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Initializing the face and eye cascade classifiers from xml files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

gaze = GazeTracking()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Directory for storing recordings
recording_directory = 'recordings'
if not os.path.exists(recording_directory):
    os.makedirs(recording_directory)

CSV_FILE = 'gaze_parameters.csv'

def enhance_image(image):
    src = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, src, -0.5, 0)

def analyze_lighting(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2]), np.std(hsv[:, :, 2])

def extract_features(faces, frame, avg_brightness, brightness_variation):
    features = {
        "eye_region_details": 0,
        "head_pose": 0,
        "head_x": 0,
        "head_y": 0,
        "head_z": 0,
        "iris_2d": 0,
        "light_intensity": avg_brightness,
        "ambient_intensity": brightness_variation
    }
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_rect = dlib.rectangle(x, y, x + w, y + h)
        shape = predictor(frame, face_rect)
        if shape:
            shape = face_utils.shape_to_np(shape)
            iris_x_mean = np.mean([pt[0] for pt in shape[36:42]])
            iris_y_mean = np.mean([pt[1] for pt in shape[36:42]])
            rect_bb = face_utils.rect_to_bb(face_rect)
            features.update({
                "iris_2d": (iris_x_mean + iris_y_mean) / 2,
                "eye_region_details": (rect_bb[0] + rect_bb[1] + rect_bb[2] + rect_bb[3]) / 4,
                "head_pose": rect_bb[2] - rect_bb[0],
                "head_x": rect_bb[0],
                "head_y": rect_bb[1],
                "head_z": rect_bb[3] - rect_bb[1]
            })

    return features

def predict_state(features):
    df = pd.DataFrame([features])
    df.fillna(0, inplace=True)
    scaled_features = scaler.transform(df)
    prediction = model.predict(scaled_features)
    state_index = np.argmax(prediction)
    state_mapping = {0: 'Stress', 1: 'Depression', 2: 'Anxiety', 3: 'Happy', 4: 'Emotional'}
    return state_mapping[state_index]

def classify_fatigue_and_stress(blink_rate, saccade_velocity, saccade_amplitude, pupil_diameter, tear_breakup_time):
    fatigue = False
    stress = False

    # Fatigue classification based on given ranges
    if blink_rate < 10 or blink_rate > 25:
        fatigue = True
    if saccade_velocity < 150 or saccade_velocity > 300:
        fatigue = True
    if saccade_amplitude < 5 or saccade_amplitude > 25:
        fatigue = True
    if tear_breakup_time < 5:
        fatigue = True

    # Stress classification based on given ranges
    if blink_rate > 25:
        stress = True
    if pupil_diameter < 2 or pupil_diameter > 8:
        stress = True

    return fatigue, stress

def classify_concentration_and_flow(fixation_duration, fixation_frequency, blink_rate, gaze_variability, saccadic_velocity):
    concentration = False
    flow = False

    # Concentration classification based on fixation duration, fixation frequency, and blink rate
    if fixation_duration > 1.0 and fixation_frequency > 1 and blink_rate < 20:
        concentration = True

    # Flow classification based on smooth eye movements and consistent gaze
    if gaze_variability[0] < 0.05 and gaze_variability[1] < 0.05 and saccadic_velocity < 200:
        flow = True

    return concentration, flow

def recommend_exercise(fatigue, stress, concentration, flow):
    if fatigue:
        print("You seem fatigued. Please try the following exercise: \n- Close your eyes and take deep breaths for 2 minutes.")
    if stress:
        print("You seem stressed. Please try the following exercise: \n- Stand up, stretch, and take a short walk for 5 minutes.")
    if not concentration:
        print("You seem to have low concentration. Please try the following exercise: \n- Focus on an object and try to maintain your attention for 2 minutes.")
    if not flow:
        print("You seem to be out of flow. Please try the following exercise: \n- Engage in an activity you enjoy for a few minutes to regain your flow state.")
    if not fatigue and not stress and concentration and flow:
        print("You are doing well! Keep up the good work.")

def measure_effectiveness(before, after):
    print("Changes in Eye-Gaze Parameters:")
    for key in before:
        if key in after:
            try:
                if isinstance(before[key], (tuple, list)):
                    before_value = np.mean(before[key])
                else:
                    before_value = float(before[key])
                if isinstance(after[key], (tuple, list)):
                    after_value = np.mean(after[key])
                else:
                    after_value = float(after[key])
                change = after_value - before_value
                print(f"{key}: Before = {before_value}, After = {after_value}, Change = {change}")
            except ValueError:
                print(f"{key}: Before = {before[key]}, After = {after[key]}")

def capture_parameters(blink_count, total_fixation_duration, fixation_count, saccade_count, fixation_points, start_time):
    blink_rate = blink_count / ((time.time() - start_time) / 60)
    fixation_duration = total_fixation_duration
    fixation_frequency = fixation_count
    saccadic_velocity = saccade_count / total_fixation_duration if total_fixation_duration > 0 else 0
    gaze_variability = (np.std([pt[0] for pt in fixation_points]), np.std([pt[1] for pt in fixation_points]))
    dispersion = np.mean([np.linalg.norm(np.array(pt) - np.mean(fixation_points, axis=0)) for pt in fixation_points])

    pupil_diameter = 3.0  # example value, replace with actual measurement
    tear_breakup_time = 15.0  # example value, replace with actual measurement

    return {
        "blink_rate": blink_rate,
        "fixation_duration": fixation_duration,
        "fixation_frequency": fixation_frequency,
        "saccadic_velocity": saccadic_velocity,
        "gaze_variability": gaze_variability,
        "dispersion": dispersion,
        "pupil_diameter": pupil_diameter,
        "tear_breakup_time": tear_breakup_time
    }

def save_parameters_to_csv(parameters, file_path):
    df = pd.DataFrame([parameters])
    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)

def load_most_recent_parameters(file_path):
    if not os.path.isfile(file_path):
        return None
    df = pd.read_csv(file_path)
    return df.iloc[-1].to_dict()

def capture_and_analyze():
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Webcam could not be opened.")
        return

    blink_count = 0
    eyes_open = True
    last_fixation_start = None
    total_fixation_duration = 0
    fixation_count = 0
    saccade_count = 0
    last_horizontal_ratio = None
    last_vertical_ratio = None
    fixation_points = []

    start_time = time.time()

    recording_filename = os.path.join(recording_directory, f'recording_{int(start_time)}.avi')
    out = cv2.VideoWriter(recording_filename, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Warning: Empty frame skipped.")
            continue

        # Write the frame to the video file
        out.write(frame)

        gaze.refresh(frame)
        frame = gaze.annotated_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 1, 1)

        # Face detection for blink analysis
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_face = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))

            if len(eyes) >= 2:
                if not eyes_open:
                    eyes_open = True
                cv2.putText(frame, "Eyes open", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                if eyes_open:
                    eyes_open = False
                    blink_count += 1
                    print("Blink detected! Total blinks: {}".format(blink_count))
                cv2.putText(frame, "Eyes closed", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if gaze.pupils_located:
            current_horizontal_ratio = gaze.horizontal_ratio()
            current_vertical_ratio = gaze.vertical_ratio()
            fixation_points.append((current_horizontal_ratio, current_vertical_ratio))

            if last_horizontal_ratio is not None and abs(current_horizontal_ratio - last_horizontal_ratio) > 0.05:
                saccade_count += 1
                print(f"Saccade Detected: {saccade_count}")

            if gaze.is_center():
                if last_fixation_start is None:
                    last_fixation_start = time.time()
                    fixation_count += 1
            else:
                if last_fixation_start is not None:
                    duration = time.time() - last_fixation_start
                    total_fixation_duration += duration
                    print(f"Fixation Duration: {duration:.2f} seconds")
                    last_fixation_start = None

            last_horizontal_ratio = current_horizontal_ratio
            last_vertical_ratio = current_vertical_ratio

        cv2.imshow("Demo", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            if last_fixation_start is not None:
                duration = time.time() - last_fixation_start
                total_fixation_duration += duration
                print(f"Fixation Duration: {duration:.2f} seconds")
                last_fixation_start = None

            frame_enhanced = enhance_image(frame)
            avg_brightness, brightness_variation = analyze_lighting(frame_enhanced)
            features = extract_features(faces, frame_enhanced, avg_brightness, brightness_variation)

            # Capture current parameters
            current_parameters = capture_parameters(blink_count, total_fixation_duration, fixation_count, saccade_count, fixation_points, start_time)

            # Load most recent parameters from CSV
            most_recent_parameters = load_most_recent_parameters(CSV_FILE)

            if most_recent_parameters:
                # Measure effectiveness of the exercise
                measure_effectiveness(most_recent_parameters, current_parameters)

            # Save current parameters to CSV
            save_parameters_to_csv(current_parameters, CSV_FILE)

            # Calculate additional parameters
            blink_rate = blink_count / ((time.time() - start_time) / 60)
            fixation_duration = total_fixation_duration
            fixation_frequency = fixation_count
            saccadic_velocity = saccade_count / total_fixation_duration if total_fixation_duration > 0 else 0
            gaze_variability = np.std([pt[0] for pt in fixation_points]), np.std([pt[1] for pt in fixation_points])
            dispersion = np.mean([np.linalg.norm(np.array(pt) - np.mean(fixation_points, axis=0)) for pt in fixation_points])

            pupil_diameter = 3.0  # example value, replace with actual measurement
            tear_breakup_time = 15.0  # example value, replace with actual measurement

            fatigue, stress = classify_fatigue_and_stress(blink_rate, saccadic_velocity, dispersion, pupil_diameter, tear_breakup_time)
            concentration, flow = classify_concentration_and_flow(fixation_duration, fixation_frequency, blink_rate, gaze_variability, saccadic_velocity)

            predicted_state = predict_state(features)
            print(f"Saccade Count: {saccade_count}")
            print(f"Total Fixation Duration: {total_fixation_duration:.2f} seconds")
            print(f"Blink Count: {blink_count}")
            print(f"Fatigue: {'Yes' if fatigue else 'No'}")
            print(f"Stress: {'Yes' if stress else 'No'}")
            print(f"Concentration: {'Yes' if concentration else 'No'}")
            print(f"Flow: {'Yes' if flow else 'No'}")

            recommend_exercise(fatigue, stress, concentration, flow)
            break

    webcam.release()
    out.release()
    cv2.destroyAllWindows()

capture_and_analyze()
