import numpy as np
import cv2

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
    def __init__(self, cam, extractor, tracker):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.extractor = extractor
        self.tracker = tracker

    def process_first_frame(self):
        self.px_ref = self.extractor.extract(self.new_frame)
        self.frame_stage = STAGE_SECOND_FRAME

    def process_second_frame(self):
        self.px_ref, self.px_cur = self.tracker.track(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur,
                                       self.px_ref,
                                       focal=self.focal,
                                       pp=self.pp,
                                       method=cv2.RANSAC,
                                       prob=0.999,
                                       threshold=1.0)

        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E,
                                                          self.px_cur,
                                                          self.px_ref,
                                                          focal=self.focal,
                                                          pp=self.pp)

        self.frame_stage = STAGE_DEFAULT_FRAME
        self.px_ref = self.px_cur

    def process_frame(self):
        self.px_ref = self.extractor.extract(self.new_frame)
        self.px_ref, self.px_cur = self.tracker.track(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur,
                                       self.px_ref,
                                       focal=self.focal,
                                       pp=self.pp,
                                       method=cv2.RANSAC,
                                       prob=0.999,
                                       threshold=1.0)

        _, R, t, mask = cv2.recoverPose(E,
                                        self.px_cur,
                                        self.px_ref,
                                        focal=self.focal,
                                        pp=self.pp)

        absolute_scale = 1
        if absolute_scale > 0.1:
            self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)
        if self.px_ref.shape[0] < kMinNumFeature:
            self.px_cur = self.extractor.extract(self.new_frame)
        self.px_ref = self.px_cur

    def update(self, img):
        assert (img.ndim == 2
                and img.shape[0] == self.cam.height
                and img.shape[1] == self.cam.width), \
            "Frame: provided image has not the same size as the camera model or image is not grayscale: {} vs ({}, {})".format(
                img.shape, self.cam.height, self.cam.width)

        self.new_frame = img
        if self.frame_stage == STAGE_DEFAULT_FRAME:
            self.process_frame()
        elif self.frame_stage == STAGE_SECOND_FRAME:
            self.process_second_frame()
        elif self.frame_stage == STAGE_FIRST_FRAME:
            self.process_first_frame()
        self.last_frame = self.new_frame
