import math

import cv2
import numpy as np


class OpticalFlowTracker:
    lk_params = dict(winSize=(21, 21),
                     # maxLevel = 3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def __init__(self, extractor):
        self.extractor = extractor

    def track(self, prev, curr):
        features = self.extractor.extract(prev)
        kp2, st, err = cv2.calcOpticalFlowPyrLK(prev, curr,
                                                prevPts=features,
                                                nextPts=None,
                                                **self.lk_params)

        st = st.reshape(st.shape[0])
        kp1 = features[st == 1]
        kp2 = kp2[st == 1]

        OpticalFlowTracker.debug(**locals())

        return kp1, kp2

    def debug(**kwargs):
        result = kwargs["prev"].copy()

        selected = zip(kwargs["kp1"], kwargs["kp2"])
        [cv2.line(result, tuple(pair[0]), tuple(pair[1]), color=(0, 255, 0), thickness=2) for pair in selected]
        cv2.imshow('Flow', result)
        cv2.waitKey(1)


class BFMatchTracker:
    orb = cv2.ORB_create()
    max_ratio = 0.75
    max_distance_ratio = 0.03

    def __init__(self, extractor):
        self.extractor = extractor
        
    def extract_keypoints(self, frame):
        extracted = self.extractor.extract(frame)
        return [cv2.KeyPoint(x=f[0], y=f[1], _size=20) for f in extracted]

    def adapt_keypoints(self, keypoints):
        return np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints])

    def select_keypoints(self, keypoints, indicies):
        return self.adapt_keypoints([keypoints[i] for i in indicies])

    def track(self, prev, curr):
        max_distance = self.max_distance_ratio * prev.shape[0]
        prev_pts = self.extract_keypoints(prev)
        prev_kps, prev_dsc = self.orb.compute(prev, prev_pts)

        curr_pts = self.extract_keypoints(curr)
        curr_kps, curr_dsc = self.orb.compute(curr, curr_pts)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(prev_dsc, curr_dsc, k=2)

        selected = []
        idx1s, idx2s = set(), set()

        filtered = []
        for m, n in matches:
            # Lowe's ratio test
            if m.distance < self.max_ratio * n.distance:
                p1 = prev_kps[m.queryIdx]
                p2 = curr_kps[m.trainIdx]

                def distance(p1, p2):
                    return math.sqrt(math.pow(p1.pt[0] - p2.pt[0], 2) +
                                     math.pow(p1.pt[1] - p2.pt[1], 2))

                # be within orb max distance
                if distance(p1, p2) < max_distance:
                    # keep around indices
                    if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
                        idx1s.add(m.queryIdx)
                        idx2s.add(m.trainIdx)
                        selected.append((p1, p2))
                        filtered.append([m])

        BFMatchTracker.debug(**locals())

        return self.adapt_keypoints(np.array(selected)[:, 0]), \
               self.adapt_keypoints(np.array(selected)[:, 1])

    def debug(**kwargs):
        result = kwargs["prev"].copy()
        def adapt_kp(kp):
            return int(kp.pt[0]), int(kp.pt[1])

        [cv2.line(result, adapt_kp(pair[0]), adapt_kp(pair[1]), color=(0, 255, 0), thickness=2) for pair in kwargs["selected"]]
        cv2.imshow('Flow', result)
        cv2.waitKey(1)
