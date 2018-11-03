from extraction.deep_extractor import *
from odometry.visual_odometry import *
from tracking.tracker import *
from visualization.plot import Plot

cap = cv2.VideoCapture('videos/driving-day-curve.mp4')

height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2

bottom_crop = 40
cam = PinholeCamera(width, height - bottom_crop, center_y=height / 2 - bottom_crop)

extractor = FeatureExtractor()
tracker = OpticalFlowTracker(extractor)
vo = VisualOdometry(cam, tracker)

display = Plot()

traj = []
while True:
    ret, frame = cap.read()
    frame = frame[:-2*bottom_crop, :]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    vo.update(img)

    cur_t = vo.cur_t
    traj.append(cur_t)

    display.update(np.array(traj))

