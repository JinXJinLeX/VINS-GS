/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 * and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "initial_alignment.h"

void solveGyroscopeBias(map<double, ImageFrame>& all_image_frame,
                        Vector3d* Bgs) {
  Matrix3d A;
  Vector3d b;
  Vector3d delta_bg;
  A.setZero();
  b.setZero();
  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  for (frame_i = all_image_frame.begin();
       next(frame_i) != all_image_frame.end(); frame_i++) {
    frame_j = next(frame_i);
    MatrixXd tmp_A(3, 3);
    tmp_A.setZero();
    VectorXd tmp_b(3);
    tmp_b.setZero();
    Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
    tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(
        O_R, O_BG);
    tmp_b =
        2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
    A += tmp_A.transpose() * tmp_A;
    b += tmp_A.transpose() * tmp_b;
  }
  delta_bg = A.ldlt().solve(b);
  ROS_WARN_STREAM("gyroscope bias initial calibration "
                  << delta_bg.transpose());

  for (int i = 0; i <= WINDOW_SIZE; i++) Bgs[i] += delta_bg;

  for (frame_i = all_image_frame.begin();
       next(frame_i) != all_image_frame.end(); frame_i++) {
    frame_j = next(frame_i);
    frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
  }
}

MatrixXd TangentBasis(Vector3d& g0) {
  Vector3d b, c;
  Vector3d a = g0.normalized();
  Vector3d tmp(0, 0, 1);
  if (a == tmp) tmp << 1, 0, 0;
  b = (tmp - a * (a.transpose() * tmp)).normalized();
  c = a.cross(b);
  MatrixXd bc(3, 2);
  bc.block<3, 1>(0, 0) = b;
  bc.block<3, 1>(0, 1) = c;
  return bc;
}

void RefineGravity(map<double, ImageFrame>& all_image_frame, Vector3d& g,
                   VectorXd& x, GS::GS_FEATURE* GS_pro) {
  Vector3d g0 = g.normalized() * G.norm();
  Vector3d lx, ly;
  // VectorXd x;
  int all_frame_count = all_image_frame.size();
  int n_state = all_frame_count * 3 + 2 + 1;

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  for (int k = 0; k < 4; k++) {
    MatrixXd lxly(3, 2);
    lxly = TangentBasis(g0);
    int i = 0;
    for (frame_i = all_image_frame.begin();
         next(frame_i) != all_image_frame.end(); frame_i++, i++) {
      frame_j = next(frame_i);

      MatrixXd tmp_A(6, 9);
      tmp_A.setZero();
      VectorXd tmp_b(6);
      tmp_b.setZero();

      double dt = frame_j->second.pre_integration->sum_dt;

      tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
      tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 *
                                Matrix3d::Identity() * lxly;
      tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() *
                                (frame_j->second.T - frame_i->second.T) / 100.0;
      tmp_b.block<3, 1>(0, 0) =
          frame_j->second.pre_integration->delta_p +
          frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] -
          frame_i->second.R.transpose() * dt * dt / 2 * g0;

      tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
      tmp_A.block<3, 3>(3, 3) =
          frame_i->second.R.transpose() * frame_j->second.R;
      tmp_A.block<3, 2>(3, 6) =
          frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
      tmp_b.block<3, 1>(3, 0) =
          frame_j->second.pre_integration->delta_v -
          frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;

      Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
      // cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
      // MatrixXd cov_inv = cov.inverse();
      cov_inv.setIdentity();

      MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
      VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

      A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
      b.segment<6>(i * 3) += r_b.head<6>();

      A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
      b.tail<3>() += r_b.tail<3>();

      A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
      A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
    }
    /* ======== 把先验 pose 约束加进来 ======== */
    // for (const auto& pp : gPosePriors) {
    //   auto it = all_image_frame.find(pp.first);
    //   if (it == all_image_frame.end()) continue;
    //   int idx = std::distance(all_image_frame.begin(), it);
    //   const double wp = pp.second.wp;
    //   const double wR = pp.second.wR;

    //   /* 位置 */
    //   MatrixXd Jp(3, n_state);
    //   Jp.setZero();
    //   Jp.block<3, 3>(0, idx * 3) = -Matrix3d::Identity();
    //   Jp.block<3, 1>(0, n_state - 1) = -it->second.R * TIC[0];
    //   Vector3d rp = (pp.second.p - it->second.T) * wp;
    //   A += Jp.transpose() * Jp * wp * wp;
    //   b += Jp.transpose() * rp * wp * wp;

    //   /* 姿态 */
    //   MatrixXd JR(3, n_state);
    //   JR.setZero();
    //   JR.block<3, 3>(0, idx * 3) = -Matrix3d::Identity();
    //   Matrix3d dR = pp.second.R.transpose() * it->second.R;
    //   Vector3d rR = 2 * Eigen::Quaterniond(dR).vec() * wR;
    //   A += JR.transpose() * JR * wR * wR;
    //   b += JR.transpose() * rR * wR * wR;
    // }

    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    VectorXd dg = x.segment<2>(n_state - 3);
    g0 = (g0 + lxly * dg).normalized() * G.norm();
    // double s = x(n_state - 1);
  }
  g = g0;
}

bool LinearAlignment(map<double, ImageFrame>& all_image_frame, Vector3d& g,
                     VectorXd& x, GS::GS_FEATURE* GS_pro) {
  int all_frame_count = all_image_frame.size();
  int n_state = all_frame_count * 3 + 3 + 1;

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  int i = 0;
  for (frame_i = all_image_frame.begin();
       next(frame_i) != all_image_frame.end(); frame_i++, i++) {
    frame_j = next(frame_i);

    MatrixXd tmp_A(6, 10);
    tmp_A.setZero();
    VectorXd tmp_b(6);
    tmp_b.setZero();

    double dt = frame_j->second.pre_integration->sum_dt;

    tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
    tmp_A.block<3, 3>(0, 6) =
        frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
    tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() *
                              (frame_j->second.T - frame_i->second.T) / 100.0;
    tmp_b.block<3, 1>(0, 0) =
        frame_j->second.pre_integration->delta_p +
        frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
    // cout << "delta_p   " <<
    // frame_j->second.pre_integration->delta_p.transpose() << endl;
    tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
    tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
    tmp_A.block<3, 3>(3, 6) =
        frame_i->second.R.transpose() * dt * Matrix3d::Identity();
    tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
    // cout << "delta_v   " <<
    // frame_j->second.pre_integration->delta_v.transpose() << endl;

    Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
    // cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
    // MatrixXd cov_inv = cov.inverse();
    cov_inv.setIdentity();

    MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
    VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

    A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
    b.segment<6>(i * 3) += r_b.head<6>();

    A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
    b.tail<4>() += r_b.tail<4>();

    A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
    A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    /* ======== 把先验 pose 约束加进来 ======== */
    // if(GS_pro[i].USE_GS && GS_pro[i].time!=0)
    // {          // 第 i 帧有外部先验
    //     /* --- 1. 位置约束 --- */
    //     MatrixXd Jp(3, n_state);
    //     Jp.setZero();
    //     Jp.block<3, 3>(0, i * 3) = -Matrix3d::Identity();          // 对速度增量
    //     Jp.block<3, 1>(0, n_state - 1) =
    //         -frame_i->second.R * TIC[0] / 100.0;                   // 对尺度（注意前面/100）

    //     Vector3d rp = (GS_pro[i].T - frame_i->second.T) / 100.0;   // 先验 T 与视觉 SfM T 的差
    //     A += Jp.transpose() * Jp;
    //     b += Jp.transpose() * rp;

    //     /* --- 2. 姿态约束 --- */
    //     MatrixXd JR(3, n_state);
    //     JR.setZero();
    //     JR.block<3, 3>(0, i * 3) = -Matrix3d::Identity();          // 对速度增量（小角度）

    //     Matrix3d dR = GS_pro[i].R.transpose() * frame_i->second.R;
    //     Vector3d rR = 2 * Eigen::Quaterniond(dR).vec();            // 四元数误差矢量部分
    //     A += JR.transpose() * JR;
    //     b += JR.transpose() * rR;
    // }
  }
  A = A * 1000.0;
  b = b * 1000.0;
  x = A.ldlt().solve(b);
  double s = x(n_state - 1) / 100.0;
  ROS_DEBUG("estimated scale: %f", s);
  g = x.segment<3>(n_state - 4);
  ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
  if (fabs(g.norm() - G.norm()) > 0.5 || s < 0) {
    return false;
  }

  RefineGravity(all_image_frame, g, x, GS_pro);
  s = (x.tail<1>())(0) / 100.0;
  (x.tail<1>())(0) = s;
  ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
  if (s < 0.0)
    return false;
  else
    return true;
}

bool VisualIMUAlignment(map<double, ImageFrame>& all_image_frame, Vector3d* Bgs,
                        Vector3d& g, VectorXd& x, GS::GS_FEATURE* GS_pro) {
  solveGyroscopeBias(all_image_frame, Bgs);

  if (LinearAlignment(all_image_frame, g, x, GS_pro))
    return true;
  else
    return false;
}
