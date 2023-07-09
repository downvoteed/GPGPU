#pragma once

#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace color_helper {

using color_vector = std::vector<uint8_t>;
using color_vectors = std::vector<color_vector>;
using similarity_vector = std::vector<float>;
using similarity_vectors = std::vector<similarity_vector>;

/**
 * Extract the color components (R/G) from a frame at position c, r
 * @param frame The OpenCV frame
 * @param c The c coordinate
 * @param r The r coordinate
 * @param r_component The R component
 * @param g_component The G component
 */
void convert(const cv::Mat& frame, const int c, const int r,
    uint8_t& r_component, uint8_t& g_component);

/**
 * Compare the color components of two frames at position c, r
 * @param bg_colors The background color components
 * @param frame2 The second OpenCV frame
 * @param c The c coordinate
 * @param r The r coordinate
 * @param r_ratio The R color similarity
 * @param g_ratio The G color similarity
 */
void compare(const color_vectors& bg_colors, const cv::Mat& frame2, const int c,
    const int r, float& r_ratio, float& g_ratio);

} // namespace color_helper