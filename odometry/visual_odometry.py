import math

import cv2
import numpy as np
from scipy.linalg import expm, norm

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500


class PinholeCamera:
    def __init__(self, width, height, f=None, center_x=None, center_y=None):
        self.width = width
        self.height = height
        self.f = f or 0.8 * width  # Estimated from standard webcam
        self.cx = center_x or width / 2
        self.cy = center_y or height / 2


class VisualOdometry:
    def __init__(self, cam, tracker):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.f
        self.pp = (cam.cx, cam.cy)
        self.tracker = tracker

    def process_frame(self):
        self.px_ref, self.px_cur = self.tracker.track(self.last_frame, self.new_frame)
        E, mask = cv2.findEssentialMat(self.px_cur,
                                       self.px_ref,
                                       focal=self.focal,
                                       pp=self.pp,
                                       method=cv2.RANSAC,
                                       prob=0.999,
                                       threshold=0.5)

        _, R, t, mask = cv2.recoverPose(E,
                                        self.px_cur,
                                        self.px_ref,
                                        focal=self.focal,
                                        pp=self.pp)

        self.cur_t = self.cur_t + self.cur_R.dot(t)
        self.cur_R = R.dot(self.cur_R)
        self.px_ref = self.px_cur

    def update(self, img):
        assert (img.ndim == 2
                and img.shape[0] == self.cam.height
                and img.shape[1] == self.cam.width), \
            "Frame: provided image has not the same size as the camera model or image is not grayscale: {} vs ({}, {})".format(
                img.shape, self.cam.height, self.cam.width)

        if self.last_frame is None:
            self.last_frame = img
            return

        self.new_frame = img
        self.process_frame()

        self.last_frame = self.new_frame
