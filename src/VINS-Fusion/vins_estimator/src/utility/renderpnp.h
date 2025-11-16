// #ifndef GS_RENDER_H
// #define GS_RENDER_H

#pragma once

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include "CameraPoseVisualization.h"
#include <eigen3/Eigen/Dense>
// #include "../estimator/estimator.h"
// #include "../estimator/parameters.h"
#include <fstream>
namespace GS
{
    class GS_RENDER
    {
    public:
        GS_RENDER();
        GS_RENDER(double t, double pose[7], cv::Mat &img1, cv::Mat &img2, cv::Mat &img3);
        double time;
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;
        cv::Mat rgb;
        cv::Mat depth;
        cv::Mat mask;
        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        bool USE_GS;
        std::vector<Eigen::Vector3d> map_points;
        // bool FIRST_GS;
    private:
    };
    class GS_RT
    {
    public:
        double time;
        Eigen::Matrix3d R;
        Eigen::Vector3d T;
    };
}

// #endif