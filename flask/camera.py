from time import time
import cv2

class Camera(object):
    """An emulated camera implementation that streams a repeated sequence of
    files 1.jpg, 2.jpg and 3.jpg at a rate of one frame per second."""

    def __init__(self):
        self.frames = [open(f + '.jpg', 'rb').read() for f in ['1', '2', '3']]
        self.cap = cv2.VideoCapture(0)
        ret,frame = self.cap.read()
        print self.frames[0]
        #self.frames[0] = frame  

    def get_frame(self):
        #ret,self.frame = self.cap.read() 
        return self.frames[int(time()) % 2]
