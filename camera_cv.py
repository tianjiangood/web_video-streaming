import time
import io
import threading
import cv2
import detect_ssd as dt 


class Camera(object):
    thread = None  # background thread that reads frames from camera
    frame = None  # current frame is stored here by background thread
    last_access = 0  # time of last client access to the camera

    def initialize(self):
        if Camera.thread is None:
            # start background frame thread
            Camera.thread = threading.Thread(target=self._thread)
            Camera.thread.start()

            # wait until frames start to be available
            while self.frame is None:
                time.sleep(0)

    def get_frame(self):
        Camera.last_access = time.time()
        self.initialize()
        return self.frame

    @classmethod
    def _thread(cls):
		frame_idx = 0
		wait_ms = 1000/1000.0
		
		video_path = "./pets3.avi" 
		#"rtsp://admin:admin12345@192.168.31.200/h264/ch1/sub/av_stream"
		cap = cv2.VideoCapture(video_path)
		
		while(1):
			# camera setup
			
			st_start = time.clock()
			
			ret,frame=cap.read()
			frame_idx = frame_idx + 1;
			if ret:
				if ((frame_idx%2) >0 ):
					continue 
				
				#frame = cv2.resize( frame,(704,576) )
				dt_rets = dt.proc_cnn_fun(frame,0.1)
				for i in range(0,len(dt_rets)):
					conf =dt_rets[i]['conf']
					x0 = int(dt_rets[i]['bd'][0]*frame.shape[1])
					y0 = int(dt_rets[i]['bd'][1]*frame.shape[0])
					x1 = int(dt_rets[i]['bd'][2]*frame.shape[1])
					y1 = int(dt_rets[i]['bd'][3]*frame.shape[0])
					label_name = dt_rets[i]['cls']
					conf = dt_rets[i]['conf']
					show_str = '%s:%.3f'%( label_name,conf )
					
					#print x0,';',y0,';',x1,';',y1
					cv2.rectangle(frame,(x0,y0),(x1,y1),(255,0,0),3)
					cv2.rectangle(frame,(x0,y0-30),(x0+120,y0),(255,255,255),-1 )
					cv2.putText(frame,show_str,(x0,y0-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)
					
				cv2.imwrite('./tmp.jpg',frame)
				cls.frame = open('./tmp.jpg').read()
			else:
				cap = cv2.VideoCapture(video_path)
				frame_idx = 0
			
				
			st_cur = time.clock()
			st_during = (st_cur-st_start)*1000
			waittime  = wait_ms-st_during
			if(waittime>0):
				cv2.waitKey(int(waittime))
			
	
	
		cls.thread = None
