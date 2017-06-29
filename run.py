#-*- coding:utf-8 -*-
#
import web
import os
import random
import sys
import numpy as np
import time

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
#os.chdir(caffe_root)
root_path = '/home/hik/workspace/justin/SSD/caffe/'
sys.path.insert(0, caffe_root+'./python')

import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

#
urls = ('/index', 'cls_upload')
render = web.template.render('./templates/')
#图像缓存目录
imgs_cache_path = './static/imgs/'

from google.protobuf import text_format
from caffe.proto import caffe_pb2
#load PASCAL VOC labels
labelmap_file = '/home/hik/workspace/justin/SSD/caffe/data/VOC0712/labelmap_voc.prototxt' #justin
labelmap_file = root_path + '/data/VOCdevkit/VOCCaltechPed/labelmap_voc.prototxt'

file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

dataset_name = 'VOCCaltechPed'
image_resize = 500

subdir = 'SSD_%dx%d'%(image_resize,image_resize)
model_direction = root_path + '/models/VGGNet/{}/{}/'.format(dataset_name,subdir)		
#model_def     = model_direction + 'deploy_web.prototxt'				               
#model_weights = model_direction + 'VGG_{}_{}_iter_80000.caffemodel'.format(dataset_name,subdir)

model_def     = root_path + '/models/VGGNet/{}/SSD_500x500/deploy_web.prototxt'.format(dataset_name)						                #justin 
model_weights = root_path + '/models/VGGNet/{}/SSD_500x500/VGG_{}_SSD_500x500_iter_80000.caffemodel'.format(dataset_name,dataset_name) #justin 

print '111111111111'
net = caffe.Net( model_def,      # defines the structure of the model
				 model_weights,  # contains the trained weights
				 caffe.TEST )     # use test mode (e.g., don't perform dropout)
print '2222'
				
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
net.blobs['data'].reshape(1,3,image_resize,image_resize)



def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames
    
	
def proc_cnn_fun(filename,thr):
	#
	ret_set = []
	image = caffe.io.load_image(filename)
	
	transformed_image = transformer.preprocess('data', image)
	net.blobs['data'].data[...] = transformed_image
	
	# Forward pass.
	st_time = time.clock()
	detections = net.forward()['detection_out']
	print 'Forward cost is %.3fs'%(time.clock()-st_time)
	
    # Parse the outputs.
	det_label= detections[0,0,:,1]
	det_conf = detections[0,0,:,2]
	det_xmin = detections[0,0,:,3]
	det_ymin = detections[0,0,:,4]
	det_xmax = detections[0,0,:,5]
	det_ymax = detections[0,0,:,6]
	
	# Get detections with confidence higher than thr.
	top_indices = [i for i, conf in enumerate(det_conf) if conf >=thr]
	top_conf = det_conf[top_indices]
	top_label_indices = det_label[top_indices].tolist()
	top_labels = get_labelname(labelmap, top_label_indices)
	top_xmin = det_xmin[top_indices]
	top_ymin = det_ymin[top_indices]
	top_xmax = det_xmax[top_indices]
	top_ymax = det_ymax[top_indices]
	
	for i in xrange(top_conf.shape[0]):
		xmin = int(round(top_xmin[i] * image.shape[1]))
		ymin = int(round(top_ymin[i] * image.shape[0]))
		xmax = int(round(top_xmax[i] * image.shape[1]))
		ymax = int(round(top_ymax[i] * image.shape[0]))
		score = top_conf[i]
		label = int(top_label_indices[i])
		label_name = top_labels[i]
		#display_txt = '%s: %.2f'%(label_name, score)
		coords = ( top_xmin[i], top_ymin[i], top_xmax[i],top_ymax[i] )
		ret_set.append( {'cls':'person','conf':score,'bd':coords} )
		
		#cv2.rectangle( image,(xmin,ymin),(xmax,ymax),color,3 )
		#cv2.putText( image,display_txt,(xmin,ymin),cv2.FONT_HERSHEY_SIMPLEX,1,(color[0],color[1],color[2],0.5),3 )
		#cv2.namedWindow("ret",0)
		#cv2.imshow("ret",image)
		#cv2.waitKey(1)
		#currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
		#currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
	
	return ret_set

#处理函数
def proc_img( img_filename ):
	print 'test img!!!'
	
	caffe.set_device(0)
	caffe.set_mode_gpu()

	cnn_ret = proc_cnn_fun(img_filename,0.3)
	print 'cnn_ret=',cnn_ret
	#结果
	ret_string = ''
	for i in range(0,len(cnn_ret)):
		conf = cnn_ret[i]['conf']
		x0 = cnn_ret[i]['bd'][0]
		y0 = cnn_ret[i]['bd'][1]
		x1 = cnn_ret[i]['bd'][2]
		y1 = cnn_ret[i]['bd'][3]
		
		ret_string += '%s %.3f %.3f %.3f %.3f %.3f '%(cnn_ret[i]['cls'],conf,x0,y0,x1,y1) 
	
	return ret_string
	
	
class cls_upload:
	def GET(self):
		web.header("Content-Type","text/html; charset=utf-8")
		return render.upload('','')
		
	def POST(self):
		#图片文件名
		save_fn = ''
		#结果文件名
		ret_fn = ''
		x = web.input(fullpath={})
		web.header("Content-Type","text/html; charset=utf-8")
		
		if 'fullpath' in x:
			in_data = x.fullpath
			if len(in_data.filename)>0:
				filepath = in_data.filename.replace('\\','/')
				filename = os.path.basename(filepath)
				save_fn = os.path.join(imgs_cache_path,filename)
				
				fout = open( save_fn,'wb')
				fout.write( in_data.file.read() ) 
				fout.close()
				
				#执行主处理
				ret_fn = proc_img( save_fn )
				
		print 'save_fn=',save_fn
		#print 'ret_fn=',ret_fn
		html_txt = render.upload(save_fn,ret_fn)
		#print html_txt
		return html_txt

if __name__ == "__main__": 
	
	app = web.application(urls, globals()) 
	app.run()
	