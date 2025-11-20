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
Eigen::Vector3d GS::GS_RENDER::reprojection(double x, double y, double z)
{
    Eigen::Matrix3d R_c2w = orientation.toRotationMatrix();
    Eigen::Vector3d t_c2w = position;
    Eigen::Vector3d P_c = Eigen::Vector3d(x,y,z);
    Eigen::Vector3d P_w = R_c2w * P_c + t_c2w;
    return P_w;
}

GS::GS_FEATURE::GS_FEATURE(GS_RENDER GS_render)
{
    R = GS_render.R;
    T = GS_render.T;
    map_pts = GS_render.map_points;
    pro_pts = GS_render.pro_points;
    time = GS_render.time;
    USE_GS = GS_render.USE_GS;
    conf_points = GS_render.conf_points;
}

GS::GS_FEATURE::GS_FEATURE()
{
    R = Eigen::Matrix3d::Ones();
    T = Eigen::Vector3d::Zero();
    time = 0;
    USE_GS = false;
}

void GS::GS_FEATURE::reset()
{
    R.setIdentity();
    T.setZero();
    time = 0.0;
    USE_GS = false;
    map_pts.clear();
    pro_pts.clear();
}