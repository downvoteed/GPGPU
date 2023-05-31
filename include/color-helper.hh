#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

namespace color_helper {

    using color_component = std::vector<uint8_t>;
    using color_components = std::vector<color_component>;
    using similarity_vector = std::vector<double>;
    using similarity_vectors = std::vector<similarity_vector>;

    /**
     * Extract the color components (R/G) from a frame
     * @param frame The OpenCV frame
     * @return The color components
     */
    color_components convert(cv::Mat frame)
    {
        color_components pixels;
        color_component r;
        color_component g;

        // Extract the color components from the frame for each pixel
        for (int i = 0; i < frame.rows; i++)
        {
            for (int j = 0; j < frame.cols; j++)
            {
                // Push the color components to the vectors
                r.push_back(frame.at<cv::Vec3b>(i, j)[2]);
                g.push_back(frame.at<cv::Vec3b>(i, j)[1]);
            }
        }

        // Push the color components to the vector of color components
        pixels.push_back(r);
        pixels.push_back(g);
        return pixels;
    }

    /**
     * Compare the color components of two frames
     * @param c1 The color components of the first frame
     * @param c2 The color components of the second frame
     * @return The color similarities
     */
    similarity_vectors compare(color_components c1, color_components c2)
    {
        similarity_vectors similarities;
        similarity_vector r;
        similarity_vector g;

        // Calculate the color similarities for each pixel
        for (int i = 0; i < c1[0].size(); i++)
        {
            // Calculate the color similarities for each color component

            uint8_t r1 = c1[0][i];
            uint8_t r2 = c2[0][i];
            uint8_t r_max = std::max(r1, r2);
            uint8_t r_min = std::min(r1, r2);
            r.push_back((double)r_min / (double)r_max);

            uint8_t g1 = c1[1][i];
            uint8_t g2 = c2[1][i];
            uint8_t g_max = std::max(g1, g2);
            uint8_t g_min = std::min(g1, g2);
            g.push_back((double)g_min / (double)g_max);
        }

        // Push the color similarities to the vector of color similarities
        similarities.push_back(r);
        similarities.push_back(g);
        return similarities;
    }

} // namespace color_helper
