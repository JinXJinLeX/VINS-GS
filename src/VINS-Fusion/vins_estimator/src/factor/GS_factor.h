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

template <typename T>
Eigen::Matrix<T, 3, 1> quaternionLog(const Eigen::Quaternion<T>& q) {
  T w = q.w();
  Eigen::Matrix<T, 3, 1> v(q.x(), q.y(), q.z());
  T norm_v = v.norm();

  if (norm_v > T(0)) {
    T theta = T(2) * ceres::acos(w);  // 使用 ceres::acos 替换 std::acos
    return theta * v / norm_v;
  } else {
    return Eigen::Matrix<T, 3, 1>::Zero();
  }
}

template <typename T>
inline void QuaternionInverse(const T q[4], T q_inverse[4]) {
  q_inverse[0] = q[0];
  q_inverse[1] = -q[1];
  q_inverse[2] = -q[2];
  q_inverse[3] = -q[3];
};

template <typename T>
T NormalizeAngle(const T& angle_degrees) {
  if (angle_degrees > T(180.0))
    return angle_degrees - T(360.0);
  else if (angle_degrees < T(-180.0))
    return angle_degrees + T(360.0);
  else
    return angle_degrees;
};

class AngleLocalParameterization {
 public:
  template <typename T>
  bool operator()(const T* theta_radians, const T* delta_theta_radians,
                  T* theta_radians_plus_delta) const {
    *theta_radians_plus_delta =
        NormalizeAngle(*theta_radians + *delta_theta_radians);

    return true;
  }

  static ceres::LocalParameterization* Create() {
    return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization,
                                                     1, 1>);
  }
};

template <typename T>
void YawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll,
                                  T R[9]) {
  T y = yaw / T(180.0) * T(M_PI);
  T p = pitch / T(180.0) * T(M_PI);
  T r = roll / T(180.0) * T(M_PI);

  R[0] = cos(y) * cos(p);
  R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
  R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
  R[3] = sin(y) * cos(p);
  R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
  R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
  R[6] = -sin(p);
  R[7] = cos(p) * sin(r);
  R[8] = cos(p) * cos(r);
};

template <typename T>
void RotationMatrixTranspose(const T R[9], T inv_R[9]) {
  inv_R[0] = R[0];
  inv_R[1] = R[3];
  inv_R[2] = R[6];
  inv_R[3] = R[1];
  inv_R[4] = R[4];
  inv_R[5] = R[7];
  inv_R[6] = R[2];
  inv_R[7] = R[5];
  inv_R[8] = R[8];
};

template <typename T>
void RotationMatrixRotatePoint(const T R[9], const T t[3], T r_t[3]) {
  r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
  r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
  r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
};

struct RelativeRTError {
  RelativeRTError(double t_x, double t_y, double t_z, double q_w, double q_x,
                  double q_y, double q_z, double t_var, double q_var)
      : t_x(t_x),
        t_y(t_y),
        t_z(t_z),
        q_w(q_w),
        q_x(q_x),
        q_y(q_y),
        q_z(q_z),
        t_var(t_var),
        q_var(q_var) {}

  template <typename T>
  bool operator()(const T* const w_q_i, const T* ti, const T* w_q_j,
                  const T* tj, T* residuals) const {
    T t_w_ij[3];
    t_w_ij[0] = tj[0] - ti[0];
    t_w_ij[1] = tj[1] - ti[1];
    t_w_ij[2] = tj[2] - ti[2];

    T i_q_w[4];
    QuaternionInverse(w_q_i, i_q_w);

    T t_i_ij[3];
    ceres::QuaternionRotatePoint(i_q_w, t_w_ij, t_i_ij);

    residuals[0] = (t_i_ij[0] - T(t_x)) / T(t_var);
    residuals[1] = (t_i_ij[1] - T(t_y)) / T(t_var);
    residuals[2] = (t_i_ij[2] - T(t_z)) / T(t_var);

    T relative_q[4];
    relative_q[0] = T(q_w);
    relative_q[1] = T(q_x);
    relative_q[2] = T(q_y);
    relative_q[3] = T(q_z);

    T q_i_j[4];
    ceres::QuaternionProduct(i_q_w, w_q_j, q_i_j);

    T relative_q_inv[4];
    QuaternionInverse(relative_q, relative_q_inv);

    T error_q[4];
    ceres::QuaternionProduct(relative_q_inv, q_i_j, error_q);

    residuals[3] = T(2) * error_q[1] / T(q_var);
    residuals[4] = T(2) * error_q[2] / T(q_var);
    residuals[5] = T(2) * error_q[3] / T(q_var);

    return true;
  }

  static ceres::CostFunction* Create(const double t_x, const double t_y,
                                     const double t_z, const double q_w,
                                     const double q_x, const double q_y,
                                     const double q_z, const double t_var,
                                     const double q_var) {
    return (new ceres::AutoDiffCostFunction<RelativeRTError, 6, 4, 3, 4, 3>(
        new RelativeRTError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, t_var, q_var)));
  }

  double t_x, t_y, t_z, t_norm;
  double q_w, q_x, q_y, q_z;
  double t_var, q_var;
};

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
    // pose[6] = q[0];
    // pose[3] = q[1];
    // pose[4] = q[2];
    // pose[5] = q[3];
  };

  template <typename T>
  bool operator()(const T* pose, T* residual) const {
    // T relative_q[4];
    // relative_q[0] = T(gs_pose[6]);
    // relative_q[1] = T(gs_pose[3]);
    // relative_q[2] = T(gs_pose[4]);
    // relative_q[3] = T(gs_pose[5]);

    // T ori_q[4];
    // ori_q[0] = q[0];
    // ori_q[1] = q[1];
    // ori_q[2] = q[2];
    // ori_q[3] = q[3];

    // T relative_q_inv[4];
    // QuaternionInverse(relative_q, relative_q_inv);

    // T error_q[4];
    // ceres::QuaternionProduct(relative_q_inv, q, error_q);

    residual[0] = (T(gs_pose[0]) - pose[0]) / T(0.1);
    residual[1] = (T(gs_pose[1]) - pose[1]) / T(0.1);
    residual[2] = (T(gs_pose[2]) - pose[2]) / T(0.2);
    // cout << gs_pose[0] <<","<< gs_pose[1] << ","<<gs_pose[2] << endl;
    // cout << pose[0] << ","<<pose[1] << ","<<pose[2] << endl;

    // residual[3] = T(2) * error_q[1] / T(0.1);
    // residual[4] = T(2) * error_q[2] / T(0.1);
    // residual[5] = T(2) * error_q[3] / T(0.1);

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
    residual[0] = (T(gs_pose[0]) - pose[0]) / T(0.001);
    residual[1] = (T(gs_pose[1]) - pose[1]) / T(0.001);
    residual[2] = (T(gs_pose[2]) - pose[2]) / T(0.005);
    // cout << gs_pose[0] << "," << gs_pose[1] << "," << gs_pose[2] << endl;
    // cout << pose[0] << "," << pose[1] << "," << pose[2] << endl;
    return true;
  }
};