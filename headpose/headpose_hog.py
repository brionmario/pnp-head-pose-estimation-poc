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
import math

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

    accumulator_pitch = 0.0
    accumulator_roll = 0.0
    accumulator_yaw = 0.0

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

            # 2D model points
            image_points = np.float32([
                (shape.part(30).x, shape.part(30).y),  # nose
                (shape.part(8).x, shape.part(8).y),  # Chin
                (shape.part(36).x, shape.part(36).y),  # Left eye left corner
                (shape.part(45).x, shape.part(45).y),  # Right eye right corner
                (shape.part(48).x, shape.part(48).y),  # Left Mouth corner
                (shape.part(54).x, shape.part(54).y),  # Right mouth corner
                (shape.part(27).x, shape.part(27).y)
            ])

            print(image_points)

            # 3D model points
            model_points = np.float32([
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (225.0, 170.0, -135.0),  # Left eye left corner
                (-225.0, 170.0, -135.0),  # Right eye right corner
                (150.0, -150.0, -125.0),  # Left Mouth corner
                (-150.0, -150.0, -125.0),  # Right mouth corner
                (0.0, 140.0, 0.0)
            ])

            # image properties. channels is not needed so _ is to drop the value
            height, width, _ = frame.shape

            # Camera internals double
            focal_length = width
            center = np.float32([width / 2, height / 2])
            camera_matrix = np.float32([[focal_length, 0.0, center[0]],
                                           [0.0, focal_length, center[1]],
                                           [0.0, 0.0, 1.0]])
            dist_coeffs = np.zeros((4, 1), dtype="float32") #Assuming no lens distortion

            retval, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

            nose_end_point3D = np.float32([[50, 0, 0],
                                      [0, 50, 0],
                                      [0, 0, 50]])

            nose_end_point2D, jacobian = cv2.projectPoints(nose_end_point3D, rvec, tvec, camera_matrix, dist_coeffs)

            rotCamerMatrix, _ = cv2.Rodrigues(rvec)

            euler_angles = getEulerAngles(rotCamerMatrix)

            # Filter angle
            accumulator_pitch = (0.5 * euler_angles[0]) + (1.0 - 0.5) * accumulator_pitch
            accumulator_yaw = (0.5 * euler_angles[1]) + (1.0 - 0.5) * accumulator_yaw
            accumulator_roll = (0.5 * euler_angles[2]) + (1.0 - 0.5) * accumulator_roll

            euler_angles[0] = accumulator_pitch
            euler_angles[1] = accumulator_yaw
            euler_angles[2] = accumulator_roll

            # TODO: Draw head angles
            # renderHeadAngles(frame, rvec, tvec, camera_matrix)

            # Draw used points for head pose estimation
            for point in image_points:
                print(point[0])
                cv2.circle(frame, (point[0], point[1]), 3, (255, 0, 255), -1)

            # Draw face angles
            pitch = "Pitch: {}".format(accumulator_pitch)
            yaw = "Yaw: {}".format(accumulator_yaw)
            roll = "Roll: {}".format(accumulator_roll)

            cv2.putText(frame, pitch, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
            cv2.putText(frame, yaw, (left, bottom + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            cv2.putText(frame, roll, (left, bottom + 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)

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


def getEulerAngles(camera_rot_matrix):
    rt = cv2.transpose(camera_rot_matrix)
    shouldBeIdentity = np.matmul(rt, camera_rot_matrix)
    identity_mat = np.eye(3,3, dtype="float32")

    isSingularMatrix = cv2.norm(identity_mat, shouldBeIdentity) < 1e-6

    euler_angles = np.float32([0.0, 0.0, 0.0])
    if not isSingularMatrix:
        return euler_angles

    sy = math.sqrt(camera_rot_matrix[0,0] * camera_rot_matrix[0,0] +  camera_rot_matrix[1,0] * camera_rot_matrix[1,0]);

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(camera_rot_matrix[2,1] , camera_rot_matrix[2,2])
        y = math.atan2(-camera_rot_matrix[2,0], sy)
        z = math.atan2(camera_rot_matrix[1,0], camera_rot_matrix[0,0])
    else:
        x = math.atan2(-camera_rot_matrix[1,2], camera_rot_matrix[1,1])
        y = math.atan2(-camera_rot_matrix[2,0], sy)
        z = 0

    x = x * 180.0 / math.pi
    y = y * 180.0 / math.pi
    z = z * 180.0 / math.pi

    euler_angles[0] = -x
    euler_angles[1] = y
    euler_angles[2] = z

    return euler_angles


if __name__ == "__main__":
    main()
