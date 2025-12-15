#pragma once
#include <ceres/ceres.h>
#include <ceres/jet.h>  // 包含支持 ceres::Jet 的数学函数
#include <ceres/rotation.h>
#include <ros/assert.h>

#include <eigen3/Eigen/Dense>
#include <iostream>

#include "../estimator/parameters.h"
#include "../utility/renderpnp.h"
#include "../utility/utility.h"

struct GSFactor {
  double pose[7];
  double gs_pose[7];

  GSFactor(double p[3], double p_[7]) {
    for (int i = 0; i < 3; i++) {
      pose[i] = p[i];
      gs_pose[i] = p_[i];
    }
    for (int i = 3; i < 7; i++) {
      gs_pose[i] = p_[i];
    }
  };

  template <typename T>
  bool operator()(const T* pose, T* residual) const {

    residual[0] = (T(gs_pose[0]) - pose[0]) / T(0.01);
    residual[1] = (T(gs_pose[1]) - pose[1]) / T(0.01);
    residual[2] = (T(gs_pose[2]) - pose[2]) / T(0.03);
    // cout << gs_pose[0] <<","<< gs_pose[1] << ","<<gs_pose[2] << endl;
    // cout << pose[0] << ","<<pose[1] << ","<<pose[2] << endl;

    Eigen::Quaternion<T> q_current(
        static_cast<T>(pose[6]), static_cast<T>(pose[3]),
        static_cast<T>(pose[4]), static_cast<T>(pose[5]));
    Eigen::Quaternion<T> q_gt(
        static_cast<T>(gs_pose[6]), static_cast<T>(gs_pose[3]),
        static_cast<T>(gs_pose[4]), static_cast<T>(gs_pose[5]));

    Eigen::Quaternion<T> q_error = q_gt.inverse() * q_current;

    Eigen::Matrix<T, 3, 1> q_error_log = quaternionLog(q_error);

    residual[3] = q_error_log[0];
    residual[4] = q_error_log[1];
    residual[5] = q_error_log[2];

    return true;
  }
};

struct GS_Factor {
  double pose[7];
  double gs_pose[7];

  GS_Factor(double p[3], double p_[7]) {
    for (int i = 0; i < 3; i++) {
      pose[i] = p[i];
      gs_pose[i] = p_[i];
    }
    for (int i = 3; i < 7; i++) {
      gs_pose[i] = p_[i];
    }
  };

  template <typename T>
  bool operator()(const T* pose, T* residual) const {
    // residual[0] = (T(gs_pose[0]) - pose[0]) / T(0.1);
    // residual[1] = (T(gs_pose[1]) - pose[1]) / T(0.1);
    // residual[2] = (T(gs_pose[2]) - pose[2]) / T(0.1);
    residual[0] = (T(gs_pose[0]) - pose[0]) / T(0.05);
    residual[1] = (T(gs_pose[1]) - pose[1]) / T(0.05);
    residual[2] = (T(gs_pose[2]) - pose[2]) / T(0.05);
    // cout << gs_pose[0] << "," << gs_pose[1] << "," << gs_pose[2] << endl;
    // cout << pose[0] << "," << pose[1] << "," << pose[2] << endl;
    return true;
  }
};

class GSProjectionFactor : public ceres::SizedCostFunction<2, 7, 7>
{
  public:
    GSProjectionFactor(const Eigen::Vector3d &_pts_3D, const Eigen::Vector3d &_pts_2D, const double &_pts_conf);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    Eigen::Vector3d pts_3D, pts_2D;
    double pts_conf;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};