#-*- coding:utf-8 -*-
import cv2
import numpy as np
import aia_face_py.aia_face as af


if __name__ == '__main__':

	normal_w = 32*30
	#初始化
	af.aia_face_py_init('/home/hik/workspace/justin/AIA_Face/demo/bin/linux64_gcc/')

	#读取图片
	img_cv = cv2.imread('./face.jpg')
	img_normal = cv2.resize(img_cv,( normal_w,img_cv.shape[0]*normal_w/img_cv.shape[1]) )  #(w,h)  #shape[0] 高度
	png = np.array(img_normal)

	#初始化class
	face_handle = af.aia_face_py() 

	#处理
	ret = face_handle.process(png)
	roi_len = len( ret['face_roi'] )


	img_w = img_normal.shape[1]
	img_h = img_normal.shape[0]

	#结果显示
	for i in range(0, roi_len ):
		roi = ret['face_roi'][i]
		cv2.rectangle(img_normal,( int(roi[1]*img_w),int(roi[2]*img_h)),( int(roi[3]*img_w),int(roi[4]*img_h)),(0,0,255),3)
		
	cv2.imwrite('./img_ret.png',img_normal)

	#cv2.imshow('ret',png_cv)
	#cv2.waitKey(0)

	print 'ret=',ret



