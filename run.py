from odometry.visual_odometry import *
from extraction.extractor import *
from tracking.tracker import *
from extraction.deep_extractor import *

height = 540
width = 960

cam = PinholeCamera(width, height, width / 2, width / 2, height / 2, height / 2)
extractor = DeepExtractor()
tracker = FeatureTracker()
vo = VisualOdometry(cam, extractor, tracker)

traj = np.zeros((600, 600, 3), dtype=np.uint8)

cap = cv2.VideoCapture('videos/test_countryroad.mp4')
while True:
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    vo.update(img)

    cur_t = vo.cur_t
    if cur_t is not None:
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
    else:
        x, y, z = 0., 0., 0.
    draw_x, draw_y = int(x) + 290, int(z) + 90
    true_x, true_y = int(vo.trueX) + 290, int(vo.trueZ) + 90

    cv2.circle(traj, (draw_x, draw_y), 1, (0, 0, 255), 1)
    cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 2)
    cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
    cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

    cv2.imshow('Road facing camera', img)
    cv2.imshow('Trajectory', traj)
    cv2.waitKey(1)
