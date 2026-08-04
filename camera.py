"""Video Feed & Capture Manager (Dev 4)"""
import cv2

class CameraManager:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.frame_count = 0

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        return ret, frame

    def release(self):
        self.cap.release()