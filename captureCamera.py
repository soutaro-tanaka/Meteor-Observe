import cv2
import os
import time
WIDTH = 864
HEIGHT = 480
FPS = 30
exp = -2
gam = 0

def save_frame_camera_key(delay=1, window_name='frame'):
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    cap.set(cv2.CAP_PROP_GAIN,200)
    cap.set(cv2.CAP_PROP_GAMMA,gam)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyWindow(window_name)


save_frame_camera_key()
