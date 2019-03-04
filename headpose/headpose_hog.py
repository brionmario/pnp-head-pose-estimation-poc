#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

This can be used to measure head pose estimation using
68 facial landmark detection along with pretrained hog and linear svm
face detection in dlib.

"""

import time
import cv2
import dlib
import numpy as np

# If True enables the verbose mode
DEBUG = True


def main():
    video_capture = cv2.VideoCapture(0)
    time.sleep(2.0)  # to give time to the camera to warm up

    if not video_capture.isOpened():
        print("Error: The camera resource is busy or unavailable")
    else:
        print("The video source has been opened correctly...")

    # Create the main window and move it
    cv2.namedWindow('Video')
    cv2.moveWindow('Video', 20, 20)

    # Declaring the face detector and landmark detector
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor('../etc/shape_predictor_68_face_landmarks.dat')

    while True:

        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Looking for faces with dlib get_frontal_face detector in the gray scale frame
        faces = face_detector(gray, 0)

        # check to see if a face was detected, and if so, draw the total
        # number of faces on the frame
        if len(faces) > 0:
            text = "{} face(s) found".format(len(faces))
            cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

        # loop over the face detections
        for face in faces:
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()

            # Drawing a green rectangle (and text) around the face.
            label_x = left
            label_y = top - 3
            if label_y < 0:
                label_y = 0
            cv2.putText(frame, "FACE", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 1)

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = landmark_detector(gray, face)
            landmark_coords = np.zeros((shape.num_parts, 2), dtype="int")

            # loop over all facial landmarks and convert them
            # to a 2-tuple of (x, y)-coordinates
            for i in range(0, shape.num_parts):
                landmark_coords[i] = (shape.part(i).x, shape.part(i).y)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw each of them
            for (i, (x, y)) in enumerate(landmark_coords):
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Showing the frame and waiting
        # for the exit command
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    video_capture.release()
    print("Bye...")


if __name__ == "__main__":
    main()
