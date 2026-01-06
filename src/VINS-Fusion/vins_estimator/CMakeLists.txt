cmake_minimum_required(VERSION 2.8.3)

project(vins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")

#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    sensor_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models
    image_transport
    gaussian_msgs)

find_package(OpenCV REQUIRED)
# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")
#for pl_net
find_package(CUDA REQUIRED)
find_package(Ceres REQUIRED)
find_package(TBB REQUIRED)
set(CUDA_PATH        "/home/seu/cuda-11.8")
set(TENSORRT_PATH    "/usr/local/TensorRT-8.5.3.1")
set(YAML_CPP_INCLUDE_DIRS "/usr/include")
set(YAML_CPP_LIBRARIES "/usr/lib/x86_64-linux-gnu/libyaml-cpp.so")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${CUDA_PATH}/targets/x86_64-linux/include
  ${TENSORRT_PATH}/include
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/thirdparty/tensorrtbuffer/include)
set(YAML_CPP_INCLUDE_DIRS "/usr/include")
set(YAML_CPP_LIBRARIES "/usr/lib/x86_64-linux-gnu/libyaml-cpp.so")
add_subdirectory(thirdparty/tensorrtbuffer) 

link_directories(
    ${CUDA_PATH}/lib64
    ${TENSORRT_PATH}/lib)
include_directories(thirdparty/tensorrtbuffer/include)
             
catkin_package()

add_library(vins_lib
    src/estimator/parameters.cpp
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
    src/estimator/gaussian_fusion.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
    src/factor/marginalization_factor.cpp
    src/factor/GS_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/utility/renderpnp.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/featureTracker/feature_tracker.cpp
    src/sp_sg/super_glue.cpp
    src/sp_sg/super_point.cpp
    src/shared_objects.cc)
target_link_libraries(vins_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    nvinfer
    nvonnxparser
    nvinfer_plugin
    cudart
    cudnn
    tensorrtbuffer
    TBB::tbb)
    
add_executable(vins_node src/rosNodeTest.cpp)
target_link_libraries(vins_node vins_lib) 

add_executable(kitti_odom_test src/KITTIOdomTest.cpp)
target_link_libraries(kitti_odom_test vins_lib) 

add_executable(kitti_gps_test src/KITTIGPSTest.cpp)
target_link_libraries(kitti_gps_test vins_lib) 

add_dependencies(vins_node gaussian_msgs_gencpp)
