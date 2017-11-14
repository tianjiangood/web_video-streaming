#-*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import copy
from aiafacepy import *

def aia_face_py_init( path ):
	ret = aia_face_init_py( path )
	return ret;
	

class aia_face_py:
	def __init__(self):
		self.handle = aia_face_cls(AIA_FACE_FUNC_MASK_DETECT)
		
	def process(self,img):
		if(self.handle):
			ret = self.handle.process(img);
			copy_ret = ret
			return copy_ret

