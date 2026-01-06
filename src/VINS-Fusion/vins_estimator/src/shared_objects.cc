#include "shared_objects.h"

std::shared_ptr<SuperPoint> SharedObjects::superpoint = nullptr;
std::shared_ptr<SuperGlue> SharedObjects::superglue   = nullptr;

cv::Mat SharedObjects::loadImage(const std::string &image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    return image;
}

void SharedObjects::initialize(const Configs &configs, const std::string &image_path) {
    superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
    superglue  = std::make_shared<SuperGlue>(configs.superglue_config);
    superpoint->build();
    superglue->build();

    // 加载图像
    cv::Mat image = loadImage(image_path);

    // 执行 SuperPoint 和 SuperGlue 操作
    Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points;
    std::vector<cv::DMatch> superglue_matches;

    superpoint->infer(image, feature_points);
    superglue->matching_points(feature_points, feature_points, superglue_matches, true);
}