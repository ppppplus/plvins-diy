#!/usr/bin/env python3
"""
本文件启动一个同名节点替代PL-VINS中的feature_tracker_node
功能是监听图像信息并使用导入的自定义点特征提取模块来进行detecting&tracking
"""
import cv2

import copy
import rospy
import torch

import numpy as np
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import ChannelFloat32
from time import time

from utils.parameter import read_image
from utils.camera_model import PinholeCamera
from linefeature_tracker import LineFeatureTracker

from sold2.model import MyLinefeatureExtractModel, MyLinefeatureMatchModel # 导入自定义线特征模型
from utils.PointTracker import PointTracker

init_pub = False
count_frame = 0

def img_callback(img_msg, linefeature_tracker):
    # 处理传入的图像，提取线特征
    # 输入：图像msg和线特征提取器
    # 输出：无返回值，发布线特征PointCloud
    global init_pub
    global count_frame

    if not init_pub :
        init_pub = True
    else :
        init_pub = False

        bridge = CvBridge()
        conver_img = bridge.imgmsg_to_cv2(img_msg, "mono8")

        # cur_img, status = read_image(conver_img, [param.height, param.width])

        # if status is False:
        #     print("Load image error, Please check image_info topic")
        #     return
        height = 120
        width = 160
        scale = 2
        linefeature_tracker.readImage(conver_img)

        if True :

            feature_lines = PointCloud()
            id_of_line = ChannelFloat32()
            u_of_endpoint = ChannelFloat32()
            v_of_endpoint = ChannelFloat32()    # u,v是线段端点
            velocity_x_of_line = ChannelFloat32()
            velocity_y_of_line = ChannelFloat32()
            feature_lines.header = img_msg.header
            feature_lines.header.frame_id = "world"

            cur_un_lines, cur_lines, ids = linefeature_tracker.undistortedLineEndPoints( scale=2 )

            for j in range(len(ids)):
                un_pts = Point32()
                un_pts.x = cur_un_lines[0,j]
                un_pts.y = cur_un_lines[1,j]
                un_pts.z = 1

                # 向建立的点云消息中加入line信息
                feature_lines.points.append(un_pts)
                id_of_line.values.append(ids[j])
                u_of_endpoint.values.append(cur_lines[0,j])
                v_of_endpoint.values.append(cur_lines[1,j])
                velocity_x_of_line.values.append(0.0)
                velocity_y_of_line.values.append(0.0)

            feature_lines.channels.append(id_of_line)
            feature_lines.channels.append(u_of_endpoint)
            feature_lines.channels.append(v_of_endpoint)
            feature_lines.channels.append(velocity_x_of_line)
            feature_lines.channels.append(velocity_y_of_line)

            pub_img.publish(feature_lines)

            ptr_toImageMsg = Image()

            ptr_toImageMsg.header = img_msg.header
            ptr_toImageMsg.height = height * scale
            ptr_toImageMsg.width = width * scale
            ptr_toImageMsg.encoding = 'bgr8'

            ptr_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")

            for pt1, pt2 in zip(cur_un_lines.T, cur_lines.T):
                pt1 = (int(round(pt1[0])), int(round(pt1[1])))
                pt2 = (int(round(pt2[0])), int(round(pt2[1])))
                # cv2.circle(ptr_image, pt2, 2, (0, 255, 0), thickness=2)
                cv2.line(ptr_image, pt1, pt2, (0, 0, 255), 2)

            ptr_toImageMsg.data = np.array(ptr_image).tostring()
            pub_match.publish(ptr_toImageMsg)



if __name__ == '__main__':

    rospy.init_node('linefeature_tracker', anonymous=False)
    yamlPath = 'config.yaml'
    my_line_extract_model = MyLinefeatureExtractModel(yamlPath)  # 利用参数文件建立自定义线特征模型
    my_line_match_model = MyLinefeatureMatchModel(nn_thresh=0.7)
    # Option_Param = readParameters()
    # print(Option_Param)

    CamearIntrinsicParam = PinholeCamera(
        fx = 461.6, fy = 460.3, cx = 363.0, cy = 248.1, 
        k1 = -2.917e-01, k2 = 8.228e-02, p1 = 5.333e-05, p2 = -1.578e-04
        )  

    #   CamearIntrinsicParam = PinholeCamera(
    #       fx = 349.199951171875, fy = 349.199951171875, cx = 322.2005615234375, cy = 246.161865234375, 
    #       k1 = -0.2870635986328125, k2 = 0.06902313232421875, p1 = 0.000362396240234375, p2 = 0.000701904296875
    #       )
    linefeature_tracker = LineFeatureTracker(my_line_extract_model, my_line_match_model, CamearIntrinsicParam) # 利用点特征模型和相机模型生成点特征处理器

    #   sub_img = rospy.Subscriber("/mynteye/left/image_color", Image, img_callback, FeatureParameters,  queue_size=100)
    sub_img = rospy.Subscriber("/cam0/image_raw", Image, img_callback, linefeature_tracker,  queue_size=100) # 监听图像，提取和追踪点特征并发布
    

    pub_img = rospy.Publisher("linefeature", PointCloud, queue_size=1000)
    pub_match = rospy.Publisher("linefeature_img", Image, queue_size=1000)

    rospy.spin()
