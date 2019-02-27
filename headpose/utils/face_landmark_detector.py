#!/usr/bin/env python

import numpy
import dlib
import os.path


class FaceLandmarkDetector:

    def __init__(self, landmarkPath):
        # Check if the file provided exist
        if (os.path.isfile(landmarkPath) == False):
            raise ValueError('haarCascade: the files specified do not exist.')

        self._predictor = dlib.shape_predictor(landmarkPath)

    ##
    # Find landmarks in the image provided
    # @param inputImg the image where the algorithm will be called
    #
    def return_landmarks(self, image, roiX, roiY, roiW, roiH, points_to_return=range(0, 68)):
        # Creating a dlib rectangle and finding the landmarks
        dlib_rectangle = dlib.rectangle(left=int(roiX), top=int(roiY), right=int(roiW), bottom=int(roiH))
        dlib_landmarks = self._predictor(image, dlib_rectangle)

        # It selects only the landmarks that
        # have been indicated in the input parameter "points_to_return".
        # It can be used in solvePnP() to estimate the 3D pose.
        self._landmarks = numpy.zeros((len(points_to_return), 2), dtype=numpy.float32)
        counter = 0
        for point in points_to_return:
            self._landmarks[counter] = [dlib_landmarks.parts()[point].x, dlib_landmarks.parts()[point].y]
            counter += 1

        return self._landmarks
