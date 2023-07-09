#pragma once

#include <boost/asio/thread_pool.hpp>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "logger.hh"
#include "segmentation-helper.hh"
#include "texture-helper.hh"

/**
 * Process the webcam stream
 * @param verbose Whether to display the log messages
 * @param num_threads The number of threads to use
 * @param learning_rate The learning_rate value for the background optimizer
 */
void process_webcam(const bool verbose, const std::optional<unsigned int> width,
                    const std::optional<unsigned int> height,
                    const bool flipped, const unsigned int num_threads,
                    const double learning_rate);