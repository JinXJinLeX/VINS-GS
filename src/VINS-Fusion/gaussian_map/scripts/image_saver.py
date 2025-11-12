#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

# 创建保存目录
save_dir = os.path.expanduser("~/frames")
os.makedirs(save_dir, exist_ok=True)

bridge = CvBridge()

def image_callback(msg):
    timestamp = msg.header.stamp.to_sec()  # 获取 ROS 时间戳（秒）
    image = bridge.imgmsg_to_cv2(msg, "bgr8")  # 转换 ROS 图像到 OpenCV 格式
    
    filename = os.path.join(save_dir, f"{timestamp:.6f}.jpg")  # 以时间戳命名
    cv2.imwrite(filename, image)
    rospy.loginfo(f"Saved image: {filename}")

rospy.init_node("image_saver", anonymous=True)
rospy.Subscriber("/camera/color/image_raw", Image, image_callback)

rospy.spin()
