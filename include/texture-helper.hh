#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

namespace texture_helper
{
    using feature_vector = std::vector<u_int8_t>;
    using similarity_vector = std::vector<double>;

    /**
     * Calculate the LBP value of a pixel in a frame
     * @param frame The OpenCV frame
     * @param x The x coordinate of the pixel
     * @param y The y coordinate of the pixel
     * @return The LBP value of the pixel
     */
    u_int8_t calculateLBP(cv::Mat frame, int x, int y)
    {
        u_int8_t lbp = 0;
        u_int8_t center = frame.at<u_int8_t>(y, x);

        // Define the relative positions of the 8 neighbors
        int neighbors[8][2] = {
            {-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}};

        // Compare the intensity value of the center pixel with its neighbors
        for (int i = 0; i < 8; i++)
        {
            int nx = x + neighbors[i][0];
            int ny = y + neighbors[i][1];

            // Check if the neighbor coordinates are within the image boundaries
            if (nx >= 0 && nx < frame.cols && ny >= 0 && ny < frame.rows)
            {
                // If the neighbor is less than the center, set the bit to 1
                u_int8_t neighbor = frame.at<u_int8_t>(ny, nx);
                lbp |= (neighbor < center) << i;
            }
        }

        return lbp;
    }

    /**
     * Extract the LBP features from a frame
     * @param frame The OpenCV frame
     * @return The LBP features
    */
    feature_vector extract(cv::Mat frame)
    {
        feature_vector features;

        // Calculate the LBP value of each pixel in the frame
        for (int i = 0; i < frame.rows; i++)
        {
            for (int j = 0; j < frame.cols; j++)
            {
                features.push_back(calculateLBP(frame, j, i));
            }
        }

        return features;
    }

    /**
     * Compare the LBP features of two frames
     * @param f1 The LBP features of the first frame
     * @param f2 The LBP features of the second frame
     * @return The similarity between the two frames
    */
    similarity_vector compare(feature_vector f1, feature_vector f2)
    {
        similarity_vector similarities;

        // Compare the LBP values of the two frames
        for (int i = 0; i < f1.size(); i++)
        {
            // Calculate the number of identical bits
            uint8_t p1 = std::popcount(f1[i]);
            uint8_t p2 = std::popcount(f2[i]);
            uint8_t identical = std::max(p1, p2);

            // Calculate the total number of bits
            uint8_t total = 8;

            // Calculate the similarity between the two LBP values as the ratio of identical bits to total bits
            similarities.push_back((double)identical / (double)total);
        }

        return similarities;
    }

} // namespace texture_helper
