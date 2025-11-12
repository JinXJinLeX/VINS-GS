#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
import math
def odometry_publisher():
    rospy.init_node('odometry_publisher', anonymous=True)

    odom_pub = rospy.Publisher('odom', Odometry, queue_size=50)
    odom_broadcaster = tf.TransformBroadcaster()

    x = 0.0
    y = 0.0
    th = 0.0

    vx = 0.1
    vy = -0.1
    vth = 0.1

    current_time = rospy.Time.now()
    last_time = rospy.Time.now()

    # 发布频率
    # rate = rospy.Rate(1.0)
    rate =rospy.Rate(0.5)

    # 添加计数器 进行测试 限制发布次数为三次
    count = 0

    while not rospy.is_shutdown() and count < 5:
        current_time = rospy.Time.now()
        # if count == 0 or count == 99:
        #     print("===============================================\n")
        #     print(current_time)
        #     print("===============================================\n")
        

        # Compute odometry based on velocities
        dt = (current_time - last_time).to_sec()
        delta_x = (vx * math.cos(th) - vy * math.sin(th)) * dt  # 使用 math.cos 和 math.sin
        delta_y = (vx * math.sin(th) + vy * math.cos(th)) * dt
        delta_th = vth * dt

        x += delta_x
        y += delta_y
        th += delta_th

        # Create a quaternion from yaw
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, th)

        # Publish the transform over tf
        odom_trans = TransformStamped()
        odom_trans.header.stamp = current_time
        odom_trans.header.frame_id = "odom"
        odom_trans.child_frame_id = "base_link"

        odom_trans.transform.translation.x = x
        odom_trans.transform.translation.y = y
        odom_trans.transform.translation.z = 0.0
        odom_trans.transform.rotation = Quaternion(*odom_quat)

        odom_broadcaster.sendTransform(
            (odom_trans.transform.translation.x, odom_trans.transform.translation.y, odom_trans.transform.translation.z),
            (odom_quat[0], odom_quat[1], odom_quat[2], odom_quat[3]),
            current_time,
            "base_link",
            "odom"
        )

        # Publish the odometry message over ROS
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom"

        # Set the position
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = Quaternion(*odom_quat)

        # Set the velocity
        odom.child_frame_id = "base_link"
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = vth

        # Publish the message
        odom_pub.publish(odom)

        last_time = current_time
        count += 1  # 每次发布后计数器加一
        rate.sleep()

if __name__ == '__main__':
    try:
        rospy.sleep(2)
        odometry_publisher()
    except rospy.ROSInterruptException:
        pass
