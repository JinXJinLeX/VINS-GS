#pragma once

// #include "config.h" // 假设您有一个 Config 类
#include "opencv2/opencv.hpp"
#include "sp_sg/super_glue.h"
#include "sp_sg/super_point.h"
#include "sp_sg/utils.h"
#include <memory>
#include <string>
#include <vector>

class SharedObjects {
public:
    static std::shared_ptr<SuperPoint> superpoint;
    static std::shared_ptr<SuperGlue> superglue;

    static void initialize(const Configs &configs, const std::string &image_path);

private:
    static cv::Mat loadImage(const std::string &image_path);
};


