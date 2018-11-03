import cv2
import numpy as np


class FeatureExtractor:
    feature = cv2.FastFeatureDetector_create(threshold=5, nonmaxSuppression=True)

    def extract(self, frame):
        features = self.feature.detect(frame)
        return np.array([x.pt for x in features], dtype=np.float32)

class GoodFeaturesExtractor:
    def extract(self, frame):
        features = cv2.goodFeaturesToTrack(frame.astype(np.uint8),
                                           3000,
                                           qualityLevel=0.05,
                                           minDistance=8)

        return np.array([f[0] for f in features])
