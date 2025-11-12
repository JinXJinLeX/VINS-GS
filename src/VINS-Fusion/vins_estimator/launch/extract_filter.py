#!/usr/bin/python

# Extract images from a bag file.

import rosbag
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

class ImageCreator():
    def __init__(self):
        self.bridge = CvBridge()
        self.frame_counter = 0  # 添加一个帧计数器

        with rosbag.Bag('/media/seu/4000098A0009885C/datasaets/M2DGR/hall_01.bag', 'r') as bag:  # 要读取的bag文件
            for topic, msg, t in bag.read_messages():
                if topic == "/camera/color/image_raw":  # 图像的topic
                    self.frame_counter += 1  # 每次读取一帧，计数器加1
                    if self.frame_counter % 3 == 1:  # 每三帧只处理第一帧
                        try:
                            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                        except CvBridgeError as e:
                            print(e)
                        timestr = "%.6f" % msg.header.stamp.to_sec()
                        image_name = timestr + ".jpg"  # 图像命名：时间戳.jpg
                        cv2.imwrite(image_name, cv_image)  # 保存图像
                        print(f"Saved image: {image_name}")  # 打印保存的图像名称

if __name__ == '__main__':
    try:
        image_creator = ImageCreator()
    except Exception as e:
        print(f"Error: {e}")
