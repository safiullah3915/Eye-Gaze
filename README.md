# Eye-Gaze

A Python project that analyzes eye features and leverages a trained model to predict a person's mental state.

## Overview

Eye-Gaze extracts key visual features of the eyes using facial landmark detection and trained machine learning to infer the user's mental or emotional state.

## Files & Components

* **`GazeTracking-master/`** – Likely contains the eye-tracking or landmark-detection logic.
* **`TrainingSet/`** – Dataset(s) used for training the predictive model.
* **`model.py`** – Core script that loads the pre-trained model and executes the inference pipeline.
* **`eye_direction_model.h5`** – Pre-trained deep learning model file.
* **`scaler.pkl`** – Preprocessing scaler (e.g. normalization parameters).
* **`shape_predictor_68_face_landmarks.dat`** – Dlib's pre-trained facial landmark detector model.
* **`requirements.txt`** – Lists Python dependencies required to run the project.

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/safiullah3915/Eye-Gaze.git
   cd Eye-Gaze
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the model**
   Modify and run `model.py` to process video or image inputs.

## Usage

`model.py` is designed to:

* Detect eye landmarks via `shape_predictor_68_face_landmarks.dat`
* Extract gaze-related features
* Preprocess features using `scaler.pkl`
* Input processed data into the `eye_direction_model.h5`
* Output inferred mental state predictions

## What You Need to Know

* The repo includes a complete packaged solution, usable out of the box.
* Input source must be provided to `model.py` (e.g., webcam feed, image file, video).
* To retrain or fine-tune the model, use the data in `TrainingSet/` (training pipeline likely within or extended from `GazeTracking-master/`).
