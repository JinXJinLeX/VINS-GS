import rosbag
import rospy
import cv2
import os
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

# ---------------- 参数 ----------------
bag_file = '/home/seu/0420_2025-04-20-12-37-44.bag'
output_txt = '/home/seu/xjl_work_space/DROID-SLAM/data/seu/parking2/timestamp_index.txt'
image_dir  = '/home/seu/xjl_work_space/DROID-SLAM/data/seu/parking2/data'
topic_name = '/camera/color/image_raw/compressed'
# -------------------------------------

os.makedirs(image_dir, exist_ok=True)
bridge = CvBridge()

timestamps = []

with rosbag.Bag(bag_file, 'r') as bag:
    for _, msg, _ in bag.read_messages(topics=[topic_name]):
        # 用 header.stamp 作为真值时间戳
        timestamp = msg.header.stamp.to_sec()
        timestamps.append(timestamp)

        # 解码保存
        cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img_path = os.path.join(image_dir, f"{timestamp:.9f}.jpg")
        cv2.imwrite(img_path, cv_img)

# 写 txt
with open(output_txt, 'w') as f:
    for ts in timestamps:
        f.write(f"{ts:.9f} {image_dir}/{ts:.9f}.jpg\n")

print(f"完成！共保存 {len(timestamps)} 张图到 ./{image_dir}/，索引文件 ./{output_txt}")
