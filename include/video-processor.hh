#pragma once

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <chrono>

#include "frame-helper.hh"
#include "segmentation-helper.hh"
#include "texture-helper.hh"

/**
 * Process a video file
 * @param verbose Whether to display the log messages
 * @param video_path The path to the video file
 * @param width The width of the frames
 * @param height The height of the frames
 * @param output_path The path to the output video file
 * @param num_threads The number of threads to use
 * @param display Whether to display the segmented frames
 * @param fps The FPS of the output video file
 * @param learning_rate The learning_rate value for the background optimizer
 */
void process_video(const bool verbose, const std::string& video_path,
                   const std::optional<unsigned int> width,
                   const std::optional<unsigned int> height,
                   const std::optional<std::string> output_path,
                   const unsigned int num_threads, const bool display,
                   const unsigned int fps, const double learning_rate);