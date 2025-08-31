# blink_detect.py
import numpy as np
import cv2

def detect_blinks():
    # Initializing the face and eye cascade classifiers from xml files
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

    # Variable to store execution state
    eyes_open = True

    # Counter for blinks
    blink_count = 0

    # Starting the video capture
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()

    while ret:
        ret, img = cap.read()
        # Converting the recorded image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Applying filter to remove impurities
        gray = cv2.bilateralFilter(gray, 5, 1, 1)

        # Detecting the face for the region of the image to be fed to the eye classifier
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # roi_face is the face which is input to the eye classifier
                roi_face = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))

                # Checking the state of the eyes
                if len(eyes) >= 2:
                    if not eyes_open:
                        eyes_open = True
                    cv2.putText(img, "Eyes open", (70, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                else:
                    if eyes_open:
                        # Transition from open to closed detected
                        eyes_open = False
                        blink_count += 1
                        print("Blink detected! Total blinks: {}".format(blink_count))
                        cv2.putText(img, "Blink detected!", (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
                        cv2.waitKey(300)
        else:
            cv2.putText(img, "No face detected", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('img', img)
        a = cv2.waitKey(1)
        if a == ord('c'):
            # Exit the loop
            print("Exiting, total blinks detected: {}".format(blink_count))
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return blink_count
