#!/usr/bin/env python3
"""
本文件定义了一个类来实现点特征提取的功能，替代PL-VINS源码中的feature_tracker.cpp
"""
import cv2
import copy
import numpy as np 
from time import time

# from utils.PointTracker import PointTracker
# from utils.feature_process import SuperPointFrontend_torch, SuperPointFrontend
run_time = 0.0
match_time = 0.0

myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])


class FeatureTracker:
	def __init__(self, extract_model, match_model, cams, min_cnt=150):
		# point_model为自定义点特征模型类，其中提供extract方法接受一个图像输入，输出特征点信息
		self.extractor = extract_model
		self.matcher = match_model
		self.forwframe_ = {
				'PointID': [],
				'keyPoint': np.zeros((3,0)),
				'descriptor': np.zeros((256,0)),
				'image': None,
				}

		self.curframe_ = {
				'PointID': [],
				'keyPoint': np.zeros((3,0)),
				'descriptor': np.zeros((256,0)),
				'image': None
				}
	
		self.camera = cams
		self.new_frame = None
		self.allfeature_cnt = 0
		self.min_cnt = min_cnt
		self.no_display = True
		
		# self.cuda = opts.cuda
		# self.scale = opts.scale
		# 
		# self.nms_dist = opts.nms_dist
		# self.nn_thresh = opts.nn_thresh
		# self.no_display = opts.no_display
		# self.width = opts.W // opts.scale
		# self.height = opts.H // opts.scale
		# self.conf_thresh = opts.conf_thresh
		# self.weights_path = opts.weights_path

		# SuperPointFrontend_torch SuperPointFrontend
		# self.SuperPoint_Ghostnet = SuperPointFrontend_torch(
		# 	weights_path = self.weights_path, 
		# 	nms_dist = self.nms_dist,
		# 	conf_thresh = self.conf_thresh,
		# 	cuda = self.cuda
		# 	)
		
		# self.tracker = PointTracker(nn_thresh=self.nn_thresh)

	def undistortedPoints(self):

		cur_un_pts = copy.deepcopy(self.curframe_['keyPoint'])
		ids = copy.deepcopy(self.curframe_['PointID'])
		cur_pts = copy.deepcopy(self.curframe_['keyPoint'])

		for i in range(cur_pts.shape[1]):
			b = self.camera.liftProjective(cur_pts[:2,i])
			cur_un_pts[0,i] = b[0] / b[2]
			cur_un_pts[1,i] = b[1] / b[2]

		return cur_un_pts, cur_pts, ids


	def readImage(self, new_img):

		# assert(new_img.ndim==2 and new_img.shape[0]==self.height and new_img.shape[1]==self.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		
		self.new_frame = new_img
		# print("wsssssssssssssssssssssssssssssssssss:", self.new_frame.ndim)
		# cv2.imshow('new_frame', self.new_frame)
		# cv2.waitKey(0)

		first_image_flag = False

		if not self.forwframe_['PointID']:
			self.forwframe_['PointID'] = []
			self.forwframe_['keyPoint'] = np.zeros((3,0))
			self.forwframe_['descriptor'] = np.zeros((256,0))

			self.forwframe_['image'] = self.new_frame
			self.curframe_['image'] = self.new_frame
			first_image_flag = True

		else:
			self.forwframe_['PointID'] = []
			self.forwframe_['keyPoint'] = np.zeros((3,0))
			self.forwframe_['descriptor'] = np.zeros((256,0))

			self.forwframe_['image'] = self.new_frame
		
		######################### 提取关键点和描述子 ############################
		print('*'*10 + " current frame " + '*'*10)
		start_time = time()
		self.forwframe_['keyPoint'], self.forwframe_['descriptor'] = self.extractor.extract(self.new_frame)

		global run_time
		run_time += ( time()-start_time )
		print("run time is :", run_time)

		num_points = self.forwframe_['keyPoint'].shape[1]
		print("current keypoint size is :", num_points)

		# if keyPoint_size < self.min_cnt-50:
		# 	self.forwframe_['keyPoint'], self.forwframe_['descriptor'], heatmap = self.SuperPoint_Ghostnet.run(self.new_frame, conf_thresh=0.01)
		# 	keyPoint_size = self.forwframe_['keyPoint'].shape[1]
		# 	print("next keypoint size is ", keyPoint_size)

		for _ in range(num_points):
			if first_image_flag == True:
				self.forwframe_['PointID'].append(self.allfeature_cnt)
				self.allfeature_cnt = self.allfeature_cnt+1
			else:
				self.forwframe_['PointID'].append(-1)
		
		##################### 开始处理匹配的特征点 ###############################
		if self.curframe_['keyPoint'].shape[1] > 0:
			start_time = time()
			matches = self.matcher.match( 
									self.forwframe_['descriptor'], 
									self.curframe_['descriptor']
							)
			# matches: [3,num_matches]
			global match_time
			match_time += time()-start_time
			print("match time is :", match_time)
			print("match size is :", matches.shape[1])
			######################## 保证匹配得到的pointID相同 #####################
			for k in range(matches.shape[1]):
				self.forwframe_['PointID'][int(matches[0,k])] = self.curframe_['PointID'][int(matches[1,k])]

			################### 将跟踪的点与没跟踪的点进行区分 #####################
			vecpoint_new = np.zeros((3,0))
			vecpoint_tracked = np.zeros((3,0))
			pointID_new = []
			pointID_tracked = []
			descr_new = np.zeros((256,0))
			descr_tracked = np.zeros((256,0))

			for i in range(num_points):
				if self.forwframe_['PointID'][i] == -1 :
					self.forwframe_['PointID'][i] = self.allfeature_cnt
					self.allfeature_cnt = self.allfeature_cnt+1
					vecpoint_new = np.append(vecpoint_new, self.forwframe_['keyPoint'][:,i:i+1], axis=1)
					pointID_new.append(self.forwframe_['PointID'][i])
					descr_new = np.append(descr_new, self.forwframe_['descriptor'][:,i:i+1], axis=1)
				else:
					vecpoint_tracked = np.append(vecpoint_tracked, self.forwframe_['keyPoint'][:,i:i+1], axis=1)
					pointID_tracked.append(self.forwframe_['PointID'][i])
					descr_tracked = np.append(descr_tracked, self.forwframe_['descriptor'][:,i:i+1], axis=1)

			########### 跟踪的点特征少于阈值了，那就补充新的点特征 ###############

			diff_n = self.min_cnt - vecpoint_tracked.shape[1]
			if diff_n > 0:
				if vecpoint_new.shape[1] >= diff_n:
					for k in range(diff_n):
						vecpoint_tracked = np.append(vecpoint_tracked, vecpoint_new[:,k:k+1], axis=1)
						pointID_tracked.append(pointID_new[k])
						descr_tracked = np.append(descr_tracked, descr_new[:,k:k+1], axis=1)
				else:
					for k in range(vecpoint_new.shape[1]):
						vecpoint_tracked = np.append(vecpoint_tracked, vecpoint_new[:,k:k+1], axis=1)
						pointID_tracked.append(pointID_new[k])
						descr_tracked = np.append(descr_tracked, descr_new[:,k:k+1], axis=1)
			
			self.forwframe_['keyPoint'] = vecpoint_tracked
			self.forwframe_['PointID'] = pointID_tracked
			self.forwframe_['descriptor'] = descr_tracked

		# if not self.no_display :	
		# 	out1 = (np.dstack((self.curframe_['image'], self.curframe_['image'], self.curframe_['image'])) * 255.).astype('uint8')
		# 	for i in range(len(self.curframe_['PointID'])):
		# 		pts1 = (int(round(self.curframe_['keyPoint'][0,i]))-3, int(round(self.curframe_['keyPoint'][1,i]))-3)
		# 		pts2 = (int(round(self.curframe_['keyPoint'][0,i]))+3, int(round(self.curframe_['keyPoint'][1,i]))+3)
		# 		pt2 = (int(round(self.curframe_['keyPoint'][0,i])), int(round(self.curframe_['keyPoint'][1,i])))
		# 		cv2.rectangle(out1, pts1, pts2, (0,255,0))
		# 		cv2.circle(out1, pt2, 2, (255, 0, 0), -1)
		# 		# cv2.putText(out1, str(self.curframe_['PointID'][i]), pt2, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , 0.3, (0, 0, 255), lineType=5)
		# 	cv2.putText(out1, 'pre_image Point', (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), lineType=16)

		# 	out2 = (np.dstack((self.forwframe_['image'], self.forwframe_['image'], self.forwframe_['image'])) * 255.).astype('uint8')
		# 	for i in range(len(self.forwframe_['PointID'])):
		# 		pts1 = (int(round(self.forwframe_['keyPoint'][0,i]))-3, int(round(self.forwframe_['keyPoint'][1,i]))-3)
		# 		pts2 = (int(round(self.forwframe_['keyPoint'][0,i]))+3, int(round(self.forwframe_['keyPoint'][1,i]))+3)
		# 		pt2 = (int(round(self.forwframe_['keyPoint'][0,i])), int(round(self.forwframe_['keyPoint'][1,i])))
		# 		cv2.rectangle(out2, pts1, pts2, (0,255,0))
		# 		cv2.circle(out2, pt2, 2, (0, 0, 255), -1)
		# 		# cv2.putText(out2, str(self.forwframe_['PointID'][i]), pt2, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 0, 255), lineType=5)
		# 	cv2.putText(out2, 'cur_image Point', (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), lineType=16)

		# 	min_conf = 0.001
		# 	heatmap[heatmap < min_conf] = min_conf
		# 	heatmap = -np.log(heatmap)
		# 	heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
		# 	out3 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
		# 	out3 = (out3*255).astype('uint8')
		# 	# print()
		# 	print(out1.shape, out2.shape, out3.shape)
		# 	out = np.hstack((out1, out2, out3))
			
		# 	out = cv2.resize(out, (3*self.width, self.height))

		# 	cv2.namedWindow("feature detector window",1)
		# 	# cv2.resizeWindow("feature detector window", 640*3, 480)
		# 	cv2.imshow('feature detector window',out)
		# 	cv2.waitKey(1)

		self.curframe_ = copy.deepcopy(self.forwframe_)

