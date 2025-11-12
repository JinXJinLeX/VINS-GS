/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "feature_tracker.h"

bool FeatureTracker::inBorder(const cv::Point2f &pt) {
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE &&
         BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

double distance(cv::Point2f pt1, cv::Point2f pt2) {
  // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
  double dx = pt1.x - pt2.x;
  double dy = pt1.y - pt2.y;
  return sqrt(dx * dx + dy * dy);
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i]) v[j++] = v[i];
  v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i]) v[j++] = v[i];
  v.resize(j);
}

FeatureTracker::FeatureTracker() {
  stereo_cam = 0;
  n_id = 0;
  hasPrediction = false;
}

void FeatureTracker::setMask() {
  mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

  // prefer to keep features that are tracked for long time
  vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

  for (unsigned int i = 0; i < cur_pts.size(); i++)
    cnt_pts_id.push_back(
        make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

  sort(cnt_pts_id.begin(), cnt_pts_id.end(),
       [](const pair<int, pair<cv::Point2f, int>> &a,
          const pair<int, pair<cv::Point2f, int>> &b) {
         return a.first > b.first;
       });

  cur_pts.clear();
  ids.clear();
  track_cnt.clear();

  for (auto &it : cnt_pts_id) {
    if (mask.at<uchar>(it.second.first) == 255) {
      cur_pts.push_back(it.second.first);
      ids.push_back(it.second.second);
      track_cnt.push_back(it.first);
      cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
    }
  }
}

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2) {
  // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
  double dx = pt1.x - pt2.x;
  double dy = pt1.y - pt2.y;
  return sqrt(dx * dx + dy * dy);
}

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
FeatureTracker::trackImage(GS::GS_RENDER &render, double _cur_time,
                           const cv::Mat &_img, const cv::Mat &_img1) {
  TicToc t_r;
  cur_time = _cur_time;
  cur_img = _img;
  row = cur_img.rows;
  col = cur_img.cols;
  cv::Mat rightImg = _img1;
  cv::Mat renderImg;
  /*
  {
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
      clahe->apply(cur_img, cur_img);
      if(!rightImg.empty())
          clahe->apply(rightImg, rightImg);
  }
  */
  cur_pts.clear();

  if (prev_pts.size() > 0) {
    TicToc t_o;
    vector<uchar> status;
    vector<float> err;
    if (hasPrediction) {
      cur_pts = predict_pts;
      cv::calcOpticalFlowPyrLK(
          prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21),
          1,
          cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                           0.01),
          cv::OPTFLOW_USE_INITIAL_FLOW);

      int succ_num = 0;
      for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) succ_num++;
      }
      if (succ_num < 10)
        cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status,
                                 err, cv::Size(21, 21), 3);
    } else
      cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status,
                               err, cv::Size(21, 21), 3);
    // reverse check
    if (FLOW_BACK) {
      vector<uchar> reverse_status;
      vector<cv::Point2f> reverse_pts = prev_pts;
      cv::calcOpticalFlowPyrLK(
          cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err,
          cv::Size(21, 21), 1,
          cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                           0.01),
          cv::OPTFLOW_USE_INITIAL_FLOW);
      // cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts,
      // reverse_status, err, cv::Size(21, 21), 3);
      for (size_t i = 0; i < status.size(); i++) {
        if (status[i] && reverse_status[i] &&
            distance(prev_pts[i], reverse_pts[i]) <= 0.5) {
          status[i] = 1;
        } else
          status[i] = 0;
      }
    }

    for (int i = 0; i < int(cur_pts.size()); i++)
      if (status[i] && !inBorder(cur_pts[i])) status[i] = 0;
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    // printf("track cnt %d\n", (int)ids.size());
  }

  for (auto &n : track_cnt) n++;

  if (1) {
    // rejectWithF();
    ROS_DEBUG("set mask begins");
    TicToc t_m;
    setMask();
    ROS_DEBUG("set mask costs %fms", t_m.toc());

    ROS_DEBUG("detect feature begins");
    TicToc t_t;
    int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
    if (n_max_cnt > 0) {
      if (mask.empty()) cout << "mask is empty " << endl;
      if (mask.type() != CV_8UC1) cout << "mask type wrong " << endl;
      cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cur_pts.size(), 0.01,
                              MIN_DIST, mask);
    } else
      n_pts.clear();
    ROS_DEBUG("detect feature costs: %f ms", t_t.toc());

    for (auto &p : n_pts) {
      cur_pts.push_back(p);
      ids.push_back(n_id++);
      track_cnt.push_back(1);
    }
    // printf("feature cnt after add %d\n", (int)ids.size());
  }

  cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
  pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

  if (!_img1.empty() && stereo_cam) {
    ids_right.clear();
    cur_right_pts.clear();
    cur_un_right_pts.clear();
    right_pts_velocity.clear();
    cur_un_right_pts_map.clear();
    if (!cur_pts.empty()) {
      // printf("stereo image; track feature on right image\n");
      vector<cv::Point2f> reverseLeftPts;
      vector<uchar> status, statusRightLeft;
      vector<float> err;
      // cur left ---- cur right
      cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts,
                               status, err, cv::Size(21, 21), 3);
      // reverse check cur right ---- cur left
      if (FLOW_BACK) {
        cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts,
                                 reverseLeftPts, statusRightLeft, err,
                                 cv::Size(21, 21), 3);
        for (size_t i = 0; i < status.size(); i++) {
          if (status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) &&
              distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
            status[i] = 1;
          else
            status[i] = 0;
        }
      }

      ids_right = ids;
      reduceVector(cur_right_pts, status);
      reduceVector(ids_right, status);
      // only keep left-right pts
      /*
      reduceVector(cur_pts, status);
      reduceVector(ids, status);
      reduceVector(track_cnt, status);
      reduceVector(cur_un_pts, status);
      reduceVector(pts_velocity, status);
      */
      cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
      right_pts_velocity =
          ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map,
                      prev_un_right_pts_map);
    }
    prev_un_right_pts_map = cur_un_right_pts_map;
  }

  // xjl
  if (!_img1.empty() && !stereo_cam) {
    cv::cvtColor(_img1, renderImg, cv::COLOR_BGR2GRAY);
    pnp_pts.clear();
    pnp_3d_pts.clear();
    cur_render_pts.clear();
    cv::Ptr<cv::ORB> orb =
        cv::ORB::create(600,   // nfeatures: 最大特征点数
                        1.3f,  // scaleFactor: 图像金字塔的尺度因子
                        9,    // nlevels: 金字塔层数
                        15,    // edgeThreshold: 边缘阈值
                        0,     // firstLevel: 金字塔的首层索引
                        2,     // WTA_K: 每对特征点之间的比较数
                        cv::ORB::HARRIS_SCORE,  // scoreType: 特征评分方法
                        31  // patchSize: 每个特征点的像素补丁大小
        );

    // 关键点和描述符
    vector<cv::KeyPoint> keypoints_cur, keypoints_render;
    vector<cv::Point3f> render_points3D;
    vector<cv::Point2f> render_points2D, cur_points2D;
    cv::Mat R, T;
    cv::Mat descriptors_cur, descriptors_render;

    // 提取 ORB 特征
    orb->detectAndCompute(cur_img, cv::Mat(), keypoints_cur, descriptors_cur);
    orb->detectAndCompute(renderImg, cv::Mat(), keypoints_render,
                          descriptors_render);

    if (!keypoints_cur.empty() && !keypoints_render.empty()) {
      // 匹配器
      cv::BFMatcher matcher(cv::NORM_HAMMING,
                            true);  // 使用汉明距离，开启交叉检查
      vector<cv::DMatch> matches;
      matcher.match(descriptors_cur, descriptors_render, matches);

      // 只保留优秀匹配
      vector<cv::DMatch> good_matches;
      for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i].distance <= 25)  // 30.0 是经验值
        {
          good_matches.push_back(matches[i]);
        }
      }

      // 提取匹配点坐标
      // cur_render_pts.clear();
      cur_pts_matched.clear();
      for (size_t i = 0; i < good_matches.size(); i++) {
        cv::Point2f pt_cur = keypoints_cur[good_matches[i].queryIdx].pt;
        cv::Point2f pt_render = keypoints_render[good_matches[i].trainIdx].pt;
        cur_points2D.push_back(pt_cur);
        render_points2D.push_back(pt_render);
      }
      // 绘制匹配结果
      cv::Mat matchImg;
      cv::drawMatches(cur_img, keypoints_cur, renderImg, keypoints_render,
                      good_matches, matchImg, cv::Scalar::all(-1),
                      cv::Scalar::all(-1), vector<char>(),
                      cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
      cv::imshow("ORB Matches", matchImg);
      cv::waitKey(1);  // 确保窗口刷新
    }
    // double cx = 324.085, cy = 232.723,
    //        fx = 603.955, fy = 603.125;// M2DGR的相机内参
    double cx = 320, cy = 240,
           fx = 613.082, fy = 526.842;// M2DGR的相机内参
    // double cx = 385.611, cy = 505.740,
    //        fx = 299.256, fy = 299.256;  // metacam的相机内参

    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    // 深度图转 3D 点
    int i = 0;
    for (const auto p : render_points2D) {
      double z = render.depth.at<float>(p.y, p.x);  // 假设深度图为 float 类型
      if ( z > 0.1 && z < 30)                        // 深度值必须大于 0
      {
        double x = z * (p.x - cx) / fx;
        double y = z * (p.y - cy) / fy;
        // printf("x: %f, y: %f, z: %f\n", x, y, z);
        render_points3D.push_back(cv::Point3f(x, y, z));
        // 将当前帧中与渲染图像匹配的关键点添加到匹配点列表中
        cur_pts_matched.push_back(cur_points2D[i]);
      }
      i++;
    }
    // PnP 求解
    std::vector<int> inliers;
    if (render_points3D.size() > 15) {
      try {
        // cv::solvePnPRansac(render_points3D, cur_pts_matched, K, cv::Mat(), R, T,
        //                    false, 150, 5.0, 0.8,
        //                    cv::Mat(),cv::SOLVEPNP_ITERATIVE);
        cv::solvePnPRansac(render_points3D, cur_pts_matched, K, cv::Mat(), R, T,
              false, 150, 5.0, 0.95, inliers, cv::SOLVEPNP_ITERATIVE);
      } catch (const cv::Exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
      }
      if (!R.empty() && !T.empty()) {
        if (R.rows == 3 && R.cols == 1) {
          cv::Mat R_matrix;
          cv::Rodrigues(R, R_matrix);  // 将旋转向量转换为旋转矩阵
          R = R_matrix;
        }

        // 计算重投影误差
        std::vector<cv::Point2f> projected_points;
        cv::projectPoints(render_points3D, R, T, K, cv::Mat(),
                          projected_points);
        double total_error = 0.0;
        for(int idx : inliers)
        {
          total_error += cv::norm(cur_pts_matched[idx] - projected_points[idx]);
        }
        double mean_error = inliers.empty() ? 1e6 : total_error / inliers.size();
        std::cout << "inliers: " << inliers.size()
          << "  mean reprojection error (inliers): " << mean_error << std::endl;
        if (mean_error < 5.0 && inliers.size()>10) {
          if (R.rows == 3 && R.cols == 1) {
            cv::Mat R_matrix;
            cv::Rodrigues(R, R_matrix);  // 如果 R 是旋转向量，先转换为旋转矩阵
            R = R_matrix;                // 更新 R 为旋转矩阵
          }
          // 转换为 Eigen::Matrix3d
          Eigen::Matrix3d eigen_R;
          for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
              eigen_R(i, j) = R.at<double>(i, j);
            }
          }
          // 转换为 Eigen::Vector3d
          Eigen::Vector3d eigen_T;
          for (int i = 0; i < 3; i++) {
            eigen_T(i) = T.at<double>(i, 0);
          }
          // 输出转换结果
          Matrix3d R_render = render.orientation.toRotationMatrix();
          Vector3d T_render = render.position;
          render.R = eigen_R.transpose() * R_render;
          render.T = R_render * (-eigen_R.transpose() * eigen_T) + T_render;
          // std::cout << "R :" << render.R << std::endl;
          // std::cout << "T :" << render.T << std::endl;
          render.USE_GS = true;
        }
      } else {
        std::cout << "R or T is empty!" << std::endl;
      }
    }
  }

  if (SHOW_TRACK)
    drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

  prev_img = cur_img;
  prev_pts = cur_pts;
  prev_un_pts = cur_un_pts;
  prev_un_pts_map = cur_un_pts_map;
  prev_time = cur_time;
  hasPrediction = false;

  prevLeftPtsMap.clear();
  for (size_t i = 0; i < cur_pts.size(); i++)
    prevLeftPtsMap[ids[i]] = cur_pts[i];

  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
  for (size_t i = 0; i < ids.size(); i++) {
    int feature_id = ids[i];
    double x, y, z;
    x = cur_un_pts[i].x;
    y = cur_un_pts[i].y;
    z = 1;
    double p_u, p_v;
    p_u = cur_pts[i].x;
    p_v = cur_pts[i].y;
    int camera_id = 0;
    double velocity_x, velocity_y;
    velocity_x = pts_velocity[i].x;
    velocity_y = pts_velocity[i].y;

    Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
    featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
  }

  if (!_img1.empty() && stereo_cam) {
    for (size_t i = 0; i < ids_right.size(); i++) {
      int feature_id = ids_right[i];
      double x, y, z;
      x = cur_un_right_pts[i].x;
      y = cur_un_right_pts[i].y;
      z = 1;
      double p_u, p_v;
      p_u = cur_right_pts[i].x;
      p_v = cur_right_pts[i].y;
      int camera_id = 1;
      double velocity_x, velocity_y;
      velocity_x = right_pts_velocity[i].x;
      velocity_y = right_pts_velocity[i].y;

      Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
      xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
      featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }
  }

  // printf("feature track whole time %f\n", t_r.toc());
  return featureFrame;
}

void FeatureTracker::rejectWithF() {
  if (cur_pts.size() >= 8) {
    ROS_DEBUG("FM ransac begins");
    TicToc t_f;
    vector<cv::Point2f> un_cur_pts(cur_pts.size()),
        un_prev_pts(prev_pts.size());
    for (unsigned int i = 0; i < cur_pts.size(); i++) {
      Eigen::Vector3d tmp_p;
      m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y),
                                  tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
      un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

      m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y),
                                  tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
      un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
    }

    vector<uchar> status;
    cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD,
                           0.99, status);
    int size_a = cur_pts.size();
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(cur_un_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(),
              1.0 * cur_pts.size() / size_a);
    ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
  }
}

void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file) {
  for (size_t i = 0; i < calib_file.size(); i++) {
    ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
    camodocal::CameraPtr camera =
        CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
    m_camera.push_back(camera);
  }
  if (calib_file.size() == 2) stereo_cam = 1;
}

void FeatureTracker::showUndistortion(const string &name) {
  cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
  vector<Eigen::Vector2d> distortedp, undistortedp;
  for (int i = 0; i < col; i++)
    for (int j = 0; j < row; j++) {
      Eigen::Vector2d a(i, j);
      Eigen::Vector3d b;
      m_camera[0]->liftProjective(a, b);
      distortedp.push_back(a);
      undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
      // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
    }
  for (int i = 0; i < int(undistortedp.size()); i++) {
    cv::Mat pp(3, 1, CV_32FC1);
    pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
    pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
    pp.at<float>(2, 0) = 1.0;
    // cout << trackerData[0].K << endl;
    // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
    // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
    if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 &&
        pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600) {
      undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300,
                               pp.at<float>(0, 0) + 300) =
          cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
    } else {
      // ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x,
      // pp.at<float>(1, 0), pp.at<float>(0, 0));
    }
  }
  // turn the following code on if you need
  // cv::imshow(name, undistortedImg);
  // cv::waitKey(0);
}

vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts,
                                                   camodocal::CameraPtr cam) {
  vector<cv::Point2f> un_pts;
  for (unsigned int i = 0; i < pts.size(); i++) {
    Eigen::Vector2d a(pts[i].x, pts[i].y);
    Eigen::Vector3d b;
    cam->liftProjective(a, b);
    un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
  }
  return un_pts;
}

vector<cv::Point2f> FeatureTracker::ptsVelocity(
    vector<int> &ids, vector<cv::Point2f> &pts,
    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts) {
  vector<cv::Point2f> pts_velocity;
  cur_id_pts.clear();
  for (unsigned int i = 0; i < ids.size(); i++) {
    cur_id_pts.insert(make_pair(ids[i], pts[i]));
  }

  // caculate points velocity
  if (!prev_id_pts.empty()) {
    double dt = cur_time - prev_time;

    for (unsigned int i = 0; i < pts.size(); i++) {
      std::map<int, cv::Point2f>::iterator it;
      it = prev_id_pts.find(ids[i]);
      if (it != prev_id_pts.end()) {
        double v_x = (pts[i].x - it->second.x) / dt;
        double v_y = (pts[i].y - it->second.y) / dt;
        pts_velocity.push_back(cv::Point2f(v_x, v_y));
      } else
        pts_velocity.push_back(cv::Point2f(0, 0));
    }
  } else {
    for (unsigned int i = 0; i < cur_pts.size(); i++) {
      pts_velocity.push_back(cv::Point2f(0, 0));
    }
  }
  return pts_velocity;
}

void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts,
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap) {
  // int rows = imLeft.rows;
  int cols = imLeft.cols;
  if (!imRight.empty() && stereo_cam)
    cv::hconcat(imLeft, imRight, imTrack);
  else
    imTrack = imLeft.clone();
  cv::cvtColor(imTrack, imTrack, cv::COLOR_GRAY2RGB);

  for (size_t j = 0; j < curLeftPts.size(); j++) {
    double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
    cv::circle(imTrack, curLeftPts[j], 2,
               cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
  }
  if (!imRight.empty() && stereo_cam) {
    for (size_t i = 0; i < curRightPts.size(); i++) {
      cv::Point2f rightPt = curRightPts[i];
      rightPt.x += cols;
      cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
      // cv::Point2f leftPt = curLeftPtsTrackRight[i];
      // cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
    }
  }

  map<int, cv::Point2f>::iterator mapIt;
  for (size_t i = 0; i < curLeftIds.size(); i++) {
    int id = curLeftIds[i];
    mapIt = prevLeftPtsMap.find(id);
    if (mapIt != prevLeftPtsMap.end()) {
      cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second,
                      cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    }
  }

  // draw prediction
  /*
  for(size_t i = 0; i < predict_pts_debug.size(); i++)
  {
      cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
  }
  */
  // printf("predict pts size %d \n", (int)predict_pts_debug.size());

  // cv::Mat imCur2Compress;
  // cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}

void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts) {
  hasPrediction = true;
  predict_pts.clear();
  predict_pts_debug.clear();
  map<int, Eigen::Vector3d>::iterator itPredict;
  for (size_t i = 0; i < ids.size(); i++) {
    // printf("prevLeftId size %d prevLeftPts size
    // %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
    int id = ids[i];
    itPredict = predictPts.find(id);
    if (itPredict != predictPts.end()) {
      Eigen::Vector2d tmp_uv;
      m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
      predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
      predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
    } else
      predict_pts.push_back(prev_pts[i]);
  }
}

void FeatureTracker::removeOutliers(set<int> &removePtsIds) {
  std::set<int>::iterator itSet;
  vector<uchar> status;
  for (size_t i = 0; i < ids.size(); i++) {
    itSet = removePtsIds.find(ids[i]);
    if (itSet != removePtsIds.end())
      status.push_back(0);
    else
      status.push_back(1);
  }

  reduceVector(prev_pts, status);
  reduceVector(ids, status);
  reduceVector(track_cnt, status);
}

cv::Mat FeatureTracker::getTrackImage() { return imTrack; }