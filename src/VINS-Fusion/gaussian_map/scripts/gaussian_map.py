#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import numpy as np
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
import rospy
sys.path.append("/home/seu/xjl_work_space/VINS-Fusion/src/splatting")
import tf.transformations
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sensor_msgs.msg import Image
import torch
import os
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
from tf import transformations as tf_trans
from functools import partial
from cv_bridge import CvBridge  # 用于转换 OpenCV 图像和 ROS Image 消息格式

from PIL import Image as Img
import torchvision.transforms as T
from gaussian_msgs.msg import image_with_pose
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
bridge = CvBridge()  # CvBridge 用于 OpenCV 和 ROS Image 消息转换
# --------------- 读取 camera.yaml ---------------
import yaml
N_ITERATION = 90000
# yaml_path = "/home/seu/xjl_work_space/VINS-Fusion/src/VINS-Fusion/config/3DGS_euroc/camera.yaml"
yaml_path = "/home/seu/xjl_work_space/VINS-Fusion/src/VINS-Fusion/config/3DGS_kasit/camera.yaml"
# yaml_path = "/home/seu/xjl_work_space/VINS-Fusion/src/VINS-Fusion/config/3DGS_312/camera.yaml"
# yaml_path = "/home/seu/xjl_work_space/VINS-Fusion/src/VINS-Fusion/config/3DGS_M2DGR/camera.yaml"
with open(yaml_path, 'r') as f:
    cfg = yaml.safe_load(f)

# 初始化 ROS Publisher
image_pub = None
index = 0  # 全局变量
# INITIAL = 0 # 渲染第一帧地图图像

width  = int(cfg['image_width'])
height = int(cfg['image_height'])
fx = cfg['projection_parameters']['fx']
fy = cfg['projection_parameters']['fy']
# 根据针孔模型计算水平/垂直视场角（单位：弧度）
fovx = 2 * np.arctan(width  / (2 * fx))
fovy = 2 * np.arctan(height / (2 * fy))
# ---------------------------------------------

# 每次接收到Odometry消息时，渲染图像 
def odometry_callback(msg, GaussianModel, pipeline, train_test_exp, separate_sh):

    global index  # 声明index为全局变量

    # custom_render_path = args.custom_render_path
    # if custom_render_path:
    #     render_path = custom_render_path
    # else:
    #     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    # # print(render_path)
    # os.makedirs(render_path, exist_ok=True)
    # 时间戳
    time = msg.header.stamp
    
    # 获取 Translation 平移向量
    translation = msg.pose.pose.position
    translation_vector = [translation.x, translation.y, translation.z]
    # print("Translation Vector:", translation_vector)

    # 获取 Rotation
    orientation = msg.pose.pose.orientation
    quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
    print("quaternion Vector:", quaternion)
    # quaternion =[-0.456027,0.269982,-0.837423,0.133686]

    
    #########################xjl M2DGR数据集转换旋转矩阵
    rotation_matrix = tf_trans.quaternion_matrix(quaternion)[:3, :3]
    # print("Rotation Matrix:", rotation_matrix)

    fb_bias = [0.0, 0.0, 0]
    bias = rotation_matrix @ fb_bias
    z_rotation_matrix = np.array([
    [1,  0,  0],
    [ 0, 1,  0],
    [ 0,  0, 1]
    ])
    new_rotation_matrix = rotation_matrix# 计算新的旋转矩阵
    rotation_matrix_transposed = new_rotation_matrix.T# 计算新的旋转矩阵（R^T）
    # print("Transposed Rotation Matrix (R^T):", rotation_matrix_transposed)
    new_translation_vector = -rotation_matrix_transposed @ translation_vector + fb_bias # 计算新的位移向量（-R^T * T）
    print("New Translation Vector:", new_translation_vector)
    # new_translation_vector = [4.40336, -3.48292, 7.93747]
    #手动设置
    resolution = (width, height)
    dummy_img = Img.new("RGB", resolution, (255, 255, 255))
    dummy_mask = np.ones(resolution, dtype=np.uint8) * 255
    # 创建 Camera 对象
    camera = Camera(
        resolution,
        colmap_id=None,
        R=rotation_matrix,
        T=new_translation_vector,
        FoVx=fovx,
        FoVy=fovy,
        depth_params=None,
        image=dummy_img,
        invdepthmap=None,
        mask=dummy_mask,
        image_name="unknown",
        uid=None,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    )
    #########################



     #########################xjl metaCam数据集转换旋转矩阵
    # rotation_matrix = tf_trans.quaternion_matrix(quaternion)[:3, :3]
    # print("Rotation Matrix:", rotation_matrix)
    # fb_bias = np.array([0.036293, -0.116754, -0.051998])
    # new_rotation_matrix = rotation_matrix# 计算新的旋转矩阵
    # rotation_matrix_transposed = new_rotation_matrix.T# 计算新的旋转矩阵（R^T）
    # print("Transposed Rotation Matrix (R^T):", rotation_matrix_transposed)
    # new_translation_vector = -rotation_matrix_transposed @ translation_vector + fb_bias # 计算新的位移向量（-R^T * T）
    # print("New Translation Vector:", new_translation_vector)
    # resolution = (width, height)                      # 760×1008 或其它
    # dummy_img = Img.new("RGB", resolution, (255, 255, 255))
    # dummy_mask = np.ones((height, width), dtype=np.uint8) * 255

    # rotation_matrix = np.array([
    #    [0.8479875, -0.4445615,  0.2885867],
    #    [0.3243531,  0.0046474, -0.9459246],
    #    [0.4191805,  0.8957363,  0.1481359 ]
    # ])
    # new_translation_vector = (0.036293, -0.116754, -0.051998)
    # camera = Camera(
    #     resolution=resolution,
    #     colmap_id=None,
    #     R=rotation_matrix,
    #     T=new_translation_vector,
    #     FoVx=fovx,
    #     FoVy=fovy,
    #     depth_params=None,
    #     image=dummy_img,
    #     invdepthmap=None,
    #     mask=dummy_mask,
    #     image_name="unknown",
    #     uid=None,
    #     trans=np.array([0.0, 0.0, 0.0]),
    #     scale=1.0,
    # )
    #########################

    # 背景颜色（需要在 GPU 上）
    bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    # 调用 render 函数进行渲染
    result = render(camera, GaussianModel, pipeline, bg_color, use_trained_exp=train_test_exp, separate_sh=separate_sh)

    # 获取渲染图像
    rendering = result["render"]
    # torchvision.utils.save_image(rendering, os.path.join('/home/seu/xjl_work_space/VINS-Fusion/rgb', '{0:05d}.png'.format(index)))
    # 将渲染图像转换为 uint8 类型，并进行 [0, 255] 归一化
    rendering_uint8 = (rendering.permute(1, 2, 0).detach().cpu().numpy() * 254).astype(np.uint8)

    # 将渲染图像转换为 ROS Image 消息
    ros_image = bridge.cv2_to_imgmsg(rendering_uint8, encoding="rgb8")
    ros_image.header.stamp = time  # 使用里程计的时间戳
    ros_image.header.frame_id = "camera" # 可选设置坐标系 ID

    depth_image = result["depth"]
    
    # 生成掩码：深度值为NaN或背景值的区域视为未渲染区域
    # depth_mask = torch.isnan(depth_image) | (depth_image <= 0.05)
    # depth_mask = depth_mask.squeeze(0).detach().cpu().numpy().astype(np.uint8) * 255  # 转为255的掩码
    # rendering_uint8[depth_mask == 255] = [0, 0, 0]  # 用黑色填充未渲染区域
    # rendering_uint8[depth_mask != 255] = [1, 1, 1]  # 用白色填充渲染区域
    # mask_ros_image = bridge.cv2_to_imgmsg(rendering_uint8, encoding="8UC3")

    # 发布渲染的深度图（原始深度值，浮点32位）
    # depth_image 是逆深度，单位 1/m
    inv_depth = depth_image.squeeze(0).detach().cpu().numpy().astype(np.float32)
    # 保护 0 值，可设成 NaN 或一个最大深度
    inv_depth[inv_depth == 0] = np.nan          # 或者 1e-4
    depth_array = 1.0 / inv_depth               # 取反：逆深度 → 深度（米）
    depth_ros_image = bridge.cv2_to_imgmsg(depth_array, encoding="32FC1")  # 使用 32 位浮点型编码
    depth_ros_image.header.stamp = time  # 使用里程计的时间戳
    depth_ros_image.header.frame_id = "camera"

    # 发布渲染的置信图（原始深度值，浮点32位）
    conf_image = result["confidence"]
    conf_array = conf_image.squeeze(0).detach().cpu().numpy().astype(np.float32)  # 转为 NumPy 浮点数组
    conf_ros_image = bridge.cv2_to_imgmsg(conf_array, encoding="32FC1")  # 使用 32 位浮点型编码
    conf_ros_image.header.stamp = time  # 使用里程计的时间戳
    conf_ros_image.header.frame_id = "camera"
    # 发布渲染的图像

    image_with_pose_msg = image_with_pose()
    image_with_pose_msg.RGBimage.header.stamp = time
    image_with_pose_msg.RGBimage = ros_image  # 设置图像
    image_with_pose_msg.DEPTHimage.header.stamp = time
    image_with_pose_msg.DEPTHimage = depth_ros_image  # 设置图像
    image_with_pose_msg.CONFimage.header.stamp = time
    image_with_pose_msg.CONFimage = conf_ros_image  # 设置图像
    # image_with_pose_msg.MASKimage.header.stamp = time
    # image_with_pose_msg.MASKimage = mask_ros_image  # 设置图像
    image_with_pose_msg.Pose.position = msg.pose.pose.position
    image_with_pose_msg.Pose.orientation = msg.pose.pose.orientation
    # image_with_pose_msg.rgbimage.append(image_with_pose_msg.RGBimage)
    # image_with_pose_msg.depthimage.append(image_with_pose_msg.DEPTHimage)
    # image_with_pose_msg.pose.append(image_with_pose_msg.Pose)

    image_with_pose_pub.publish(image_with_pose_msg)

    # 增加index值
    index += 1

def image_callback(msg):
    """
    图像回调函数
    每当接收到图像消息时，该函数会被调用
    """
    try:
        # 将ROS图像消息转换为OpenCV图像（BGR格式）
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

        # 显示图像
        cv2.imshow("Received Image", cv_image)
        cv2.waitKey(0.5)  # 等待1毫秒，确保图像窗口更新

    except CvBridgeError as e:
        rospy.logerr("CvBridgeError: {0}".format(e))

def main():
    rospy.init_node('gaussian_map', anonymous=True)
    parser = ArgumentParser(description="Testing script parameters")

    # 解析命令行参数，初始化 model, pipeline 和 args
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", default="true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--custom_render_path", default="/home/wendy/workSpace/Data/office3/output/realTimeRender", type=str, help="Custom path for rendering images")
   
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    args = get_combined_args(parser)
    
    gaussians = GaussianModel(model.extract(args).sh_degree)
    Scene(model.extract(args), gaussians, load_iteration=N_ITERATION, shuffle=False)

    # 在订阅之前初始化
    safe_state(args.quiet)
    # 使用 functools.partial 来传递额外的参数到回调函数
    odometry_callback_with_args = partial(odometry_callback, GaussianModel=gaussians, pipeline=pipeline, train_test_exp = model.train_test_exp, separate_sh = SPARSE_ADAM_AVAILABLE)
    # 订阅位姿信息的话题名称 /odom，并将回调函数传递进去  发布渲染图像
    sub = rospy.Subscriber('/vins_estimator/camera_pose', Odometry, odometry_callback_with_args, queue_size = 1000)
    # sub = rospy.Subscriber('/vins_estimator/gt_pose', Odometry, odometry_callback_with_args, queue_size = 10000)

    # 初始化图像发布器
    # global image_pub, depth_pub
    global image_with_pose_pub

    image_with_pose_pub = rospy.Publisher('/render',  image_with_pose, queue_size = 100)

    # 保持节点持续运行
    rospy.spin()


if __name__ == "__main__":
    main()