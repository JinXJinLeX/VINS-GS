cmake_minimum_required(VERSION 3.0.2)
project(gaussian_msgs)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  message_generation
  std_msgs
  sensor_msgs
  geometry_msgs
)

add_message_files(
        # DIRECTORY msg
        FILES
        image_with_pose.msg
)

generate_messages(
        DEPENDENCIES
        std_msgs
        sensor_msgs
        geometry_msgs
)

catkin_package(
CATKIN_DEPENDS roscpp rospy std_msgs message_runtime 
sensor_msgs geometry_msgs
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
