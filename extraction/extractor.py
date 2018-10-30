import cv2
import numpy as np


class FeatureExtractor:
    feature = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

    def extract(self, frame):
        features = self.feature.detect(frame)
        return np.array([x.pt for x in features], dtype=np.float32)
