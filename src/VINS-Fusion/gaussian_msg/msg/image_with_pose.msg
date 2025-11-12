# float32 time
# int64 width
# int64 height
# string encoding
# uint8 is_bigendian
# uint32 step
#rgb
# uint8[] rgbdata
#depyh
# float32[] depthdata
#odometry
# float32 x
# float32 y
# float32 z
# float32 qx
# float32 qy
# float32 qz
# float32 qw
sensor_msgs/Image RGBimage
sensor_msgs/Image DEPTHimage
sensor_msgs/Image CONFimage
geometry_msgs/Pose Pose
sensor_msgs/Image[] rgbimage
sensor_msgs/Image[] depthimage
sensor_msgs/Image[] confimage
geometry_msgs/Pose[] pose
# sensor_msgs/Image MASKimage