#pragma once

#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

namespace texture_helper
{
    using feature_vector = std::vector<uint8_t>;

    /**
     * Get the pixel value of a frame
     * @param frame The OpenCV frame
     * @param c The c coordinate of the pixel
     * @param r The r coordinate of the pixel
     * @return The pixel value
     */
    uint8_t getPixel(const cv::Mat& frame, const int c, const int r);

    /**
     * Calculate the LBP value of a pixel in a frame
     * @param frame The OpenCV frame
     * @param c The c coordinate of the pixel
     * @param r The r coordinate of the pixel
     * @return The LBP value of the pixel
     */
    uint8_t calculateLBP(const cv::Mat& frame, const int c, const int r);

    /**
     * Compare the LBP features of two frames
     * @param i The index of the LBP feature
     * @param f1 The LBP features of the first frame
     * @param f2 The LBP features of the second frame
     * @return The similarity between the two frames
     */
    float compare(const unsigned int i, const feature_vector& f1,
                  const feature_vector& f2);
} // namespace texture_helper
