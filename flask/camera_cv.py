#-*- coding:utf-8 -*-
#
import time
import io
import threading
import cv2
import numpy as np


import aia_face_py.aia_face as af
import detect_ssd as dt 


def detect_face(face_handle,img_cv):

	ret_set = []
	normal_w = 32*30
	if( normal_w<img_cv.shape[1] ):
		img_normal = cv2.resize(img_cv,( normal_w,img_cv.shape[0]*normal_w/img_cv.shape[1]) )  #(w,h)  #shape[0] 高度
	else:
		img_normal = img_cv
	png = np.array(img_normal)

	#处理
	ret = face_handle.process(png)
	roi_len = len( ret['face_roi'] )
	
	for i in xrange(roi_len ):
		roi = ret['face_roi'][i]
		coords = ( roi[1],roi[2],roi[3],roi[4] )
		ret_set.append( {'cls':'face','conf':roi[0],'bd':coords} )
		
	return ret_set


class Camera(object):
    thread = None  # background thread that reads frames from camera
    frame = None  # current frame is stored here by background thread
    last_access = 0  # time of last client access to the camera
    
    detect_thr = 0.3
    detect_method = 'ssd'# 'face,yolo,ssd'
    

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
		wait_ms = 1000/10.0
		ssd_net = 0
		
		#初始化
		af.aia_face_py_init('/home/hik/workspace/justin/AIA_Face/demo/bin/linux64_gcc/')
		
		face_handle = af.aia_face_py() 
		
		ssd_net = dt.ssd_detect_init('ssd_person_500')
		
		#video_path = "./pets3.avi"
		#video_path = "./face_test_cut.avi"
		video_path ="rtsp://admin:admin12345@192.168.0.70/h264/ch1/sub/av_stream"
		cap = cv2.VideoCapture(video_path)
		
		while(1):
			# camera setup
			
			st_start = time.clock()
			
			ret,frame=cap.read()
			frame_idx = frame_idx + 1;
			if ret:
				if ((frame_idx%4) >0 ):
					continue 
				
				#frame = cv2.resize( frame,(352,288) )
				if(cls.detect_method=='ssd' and ssd_net is not 0 ):
				        dt_rets = dt.proc_cnn_fun(ssd_net,frame,cls.detect_thr)
				elif (cls.detect_method=='face'):
				        dt_rets = detect_face( face_handle,frame)
				else:
				      dt_rets = []  
				        
				for i in xrange(len(dt_rets)):
					#print 'loop %d'%i
					x0 = int(dt_rets[i]['bd'][0]*frame.shape[1])
					y0 = int(dt_rets[i]['bd'][1]*frame.shape[0])
					x1 = int(dt_rets[i]['bd'][2]*frame.shape[1])
					y1 = int(dt_rets[i]['bd'][3]*frame.shape[0])
					label_name = dt_rets[i]['cls']
					conf = dt_rets[i]['conf']
					show_str = '%s:%.3f'%( label_name,conf )
					
					#print x0,';',y0,';',x1,';',y1
					cv2.rectangle(frame,(x0,y0),(x1,y1),(0,0,255),3)
					#cv2.rectangle(frame,(x0,y0-30),(x0+120,y0),(255,255,255),-1 )
					cv2.putText(frame,show_str,(x0,y0-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),3)
					
				cv2.imwrite('./tmp.jpg',frame)
				fp = open('./tmp.jpg')
				cls.frame = fp.read()
				fp.close()
			else:
				cap.release()
				cap = cv2.VideoCapture(video_path)
				frame_idx = 0
			
				
			st_cur = time.clock()
			st_during = (st_cur-st_start)*1000
			waittime  = wait_ms-st_during
			if(waittime>0):
				cv2.waitKey(int(waittime))
			
	
	
		cls.thread = None
