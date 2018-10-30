import cv2

class FeatureTracker:
    lk_params = dict(winSize=(21, 21),
                     # maxLevel = 3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def track(self, prev, curr, features):
        kp2, st, err = cv2.calcOpticalFlowPyrLK(prev, curr,
                                                prevPts=features,
                                                nextPts=None,
                                                **self.lk_params)

        st = st.reshape(st.shape[0])
        kp1 = features[st == 1]
        kp2 = kp2[st == 1]

        return kp1, kp2
