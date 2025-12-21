#include "GS_factor.h"

Eigen::Matrix2d GSProjectionFactor::sqrt_info=FOCAL_LENGTH / 0.5 * Eigen::Matrix2d::Identity();

GSProjectionFactor::GSProjectionFactor(const Eigen::Vector3d &_pts_3D, const Eigen::Vector3d &_pts_2D, const double &_pts_conf)
{
    pts_3D = _pts_3D;
    pts_2D = _pts_2D;
    pts_conf = _pts_conf;
};

bool GSProjectionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Vector3d P(parameters[0][0], parameters[0][1], parameters[0][2]); // 载体body
    Eigen::Quaterniond Q(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d tic(parameters[1][0], parameters[1][1], parameters[1][2]);// 外参
    Eigen::Quaterniond qic(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    // double inv_dep = parameters[2][2];                                                                       
    Eigen::Vector3d pts_imu = Q.inverse() * (pts_3D - P);
    Eigen::Vector3d pts_camera = qic.inverse() * (pts_imu - tic);
    Eigen::Map<Eigen::Vector2d> residual(residuals);

    double dep = (pts_camera.z()) + 0.001;
    residual = (pts_camera / dep).head<2>() - pts_2D.head<2>();
    // cout << "3dPoints:" << pts_camera.x() << "," << pts_camera.y() << "," << pts_camera.z()<< endl;
    // cout << "2dPoints:" << pts_2D.x() << "," << pts_2D.y() << "," << pts_2D.z() <<endl;
    // cout << "GSconf:" << pts_conf << endl;

    residual = pts_conf * sqrt_info * residual;
    // cout << "GS残差:" << residual.transpose() << endl;
    if (jacobians)
    {
        Eigen::Matrix3d R = Q.toRotationMatrix();
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix<double, 2, 3> reduce(2, 3);
        reduce.setZero();
        reduce.block<2, 3>(0,0) << 1.0 / dep, 0, -pts_camera(0) / (dep * dep),
                0, 1.0 / dep, -pts_camera(1) / (dep * dep);
        reduce = sqrt_info * reduce;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco;
            jaco.leftCols<3>() = ric.transpose() * -R.transpose();
            jaco.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu);

            jacobian_pose.leftCols<6>() = pts_conf * reduce * jaco;
            jacobian_pose.rightCols<1>().setZero();
        }
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_ex(jacobians[1]);

            Eigen::Matrix<double, 3, 6> jaco;
            jaco.leftCols<3>() = -ric.transpose();
            jaco.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_camera);

            jacobian_pose_ex.leftCols<6>() = pts_conf * reduce * jaco;
            jacobian_pose_ex.rightCols<1>().setZero();
        }
    }
    return true;
}