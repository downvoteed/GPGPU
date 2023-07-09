#pragma once

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <vector>

#include "color-helper.hh"
#include "logger.hh"
#include "texture-helper.hh"

namespace segmentation_helper
{
    const float THRESHOLD = 0.67;

    /**
     * Optimize the background model
     * @param colored_bg_frame The background frame in color
     * @param colored_frame The current frame in color
     * @param w The width of the frame
     * @param i The index of the current pixel
     * @param value The value of the current pixel
     * @param learning_rate The learning rate for adaptive alpha adjustment
     * @param weights The weights for the weighted average blending
     */
    void bg_optimization(cv::Mat* colored_bg_frame,
                         const cv::Mat& colored_frame, const unsigned int w,
                         const unsigned int i, const uint8_t value,
                         const double learning_rate, cv::Mat& weights);

    /**
     * Segment a frame into foreground and background
     * @param color_similarities The color similarities between the current
     * frame and the background frame
     * @param bg_features The texture features of the background frame
     * @param features The texture features of the current frame
     * @param w The width of the frame
     * @param result The segmented frame
     * @param colored_bg_frame The background frame in color
     * @param colored_frame The current frame in color
     * @param learning_rate The learning_rate value for the background optimizer
     */
    void segment(const color_helper::similarity_vectors& color_similarities,
                 const texture_helper::feature_vector& bg_features,
                 const texture_helper::feature_vector& features,
                 const unsigned int w, cv::Mat& result,
                 cv::Mat* colored_bg_frame, const cv::Mat& colored_frame,
                 const double learning_rate, cv::Mat& weights);

    /**
     * Segment a block of a frame based on the color and texture similarities
     * with the background frame
     * @param min_c The minimum c coordinate of the block
     * @param max_c The maximum c coordinate of the block
     * @param min_r The minimum r coordinate of the block
     * @param max_r The maximum r coordinate of the block
     * @param color_similarities The color similarities between the current
     * frame and the background frame
     * @param features The texture features of the current frame
     * @param bg_colors The background color components
     * @param colored_frame The current frame in color
     * @param gray_frame The current frame in grayscale
     * @param w The width of the frame
     */
    void segment_block(const unsigned int min_c, const unsigned int max_c,
                       const unsigned int min_r, const unsigned int max_r,
                       color_helper::similarity_vectors* color_similarities,
                       texture_helper::feature_vector* features,
                       const color_helper::color_vectors& bg_colors,
                       const cv::Mat& colored_frame, const cv::Mat& gray_frame,
                       const unsigned int w);

    /**
     * Segment a frame based on the color and texture similarities with the
     * background frame
     * @param i The index of the current frame
     * @param size The total number of frames
     * @param colored_bg_frame The background frame in color
     * @param bg_features The texture features of the background frame
     * @param bg_colors The background color components
     * @param colored_frame The current frame in color
     * @param gray_frame The current frame in grayscale
     * @param w The width of the frame
     * @param h The height of the frame
     * @param verbose Whether to display the progress
     * @param result The segmented frame
     * @param num_threads The number of threads to use
     * @param learning_rate The learning_rate value for the background optimizer
     * @param weights The weights for the weighted average blending
     */
    void segment_frame(const int i, const unsigned int size,
                       cv::Mat* colored_bg_frame,
                       const texture_helper::feature_vector& bg_features,
                       const color_helper::color_vectors& bg_colors,
                       const cv::Mat& colored_frame, const cv::Mat& gray_frame,
                       const unsigned int w, const unsigned int h,
                       const bool verbose, cv::Mat& result,
                       const unsigned int num_threads,
                       const double learning_rate, cv::Mat& weights);

    /**
     * Extract the color and texture features from the background frame
     * @param w The width of the frame
     * @param h The height of the frame
     * @param colored_bg_frame The background frame in color
     * @param bg_colors The background color components
     * @param bg_features The texture features of the background frame
     */
    void extract_frame(const unsigned int w, const unsigned int h,
                       cv::Mat* colored_bg_frame,
                       color_helper::color_vectors* bg_colors,
                       texture_helper::feature_vector* bg_features);

} // namespace segmentation_helper
