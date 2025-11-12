#include "renderpnp.h"

GS::GS_RENDER::GS_RENDER(double t, double pose[7], cv::Mat &img1, cv::Mat &img2, cv::Mat &img3)
    : time(t), rgb(img1), depth(img2), mask(img3)
{
    // time = t;
    position[0] = pose[0];
    position[1] = pose[1];
    position[2] = pose[2];
    orientation.x() = pose[3];
    orientation.y() = pose[4];
    orientation.z() = pose[5];
    orientation.w() = pose[6];
    // rgb = img1;
    // depth = img2;
}
GS::GS_RENDER::GS_RENDER()
{
    time = 0;
    position[0] = 0.0;
    position[1] = 0.0;
    position[2] = 0.0;
    orientation.x() = 0.0;
    orientation.y() = 0.0;
    orientation.z() = 0.0;
    orientation.w() = 1.0;
    R = Eigen::Matrix3d::Ones();
    T = Eigen::Vector3d::Zero();
}