#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "logger.hh"

namespace frame_helper
{
    using frames = std::vector<cv::Mat>;
    using frames_ref = std::vector<cv::Mat*>;
    using frames_vector = std::vector<frames>;

    /**
     * Read a frame from a file
     * @param path The path to the file
     * @param gray Whether to read the frame in grayscale
     * @return The OpenCV frame
     */
    const cv::Mat& readFrame(const std::string& path, const bool gray = false);

    /**
     * Show a frame in a window
     * @param name The name of the window
     * @param frame The OpenCV frame
     */
    void showFrame(const std::string& name, const cv::Mat& frame);

    /**
     * Build a segmented frame from a vector of boolean values
     * (foreground/background)
     * @param frame The frame to segment
     * @param i The index of the current value
     * @param value 0 for foreground, 1 for background
     * @param width The width of the frame
     * @return The segmented frame
     */
    const cv::Mat& buildSegmentedFrame(cv::Mat& frame, const unsigned int i,
                                       const uint8_t value, const int width);

    /**
     * Read a video file and return its frames
     * @param path The path to the video file
     * @param width The width of the frames (optional)
     * @param height The height of the frames (optional)
     * @return The frames
     */
    const frames_vector*
    readFrames(const std::string& path,
               const std::optional<int>& width = std::nullopt,
               const std::optional<int>& height = std::nullopt);

    /**
     * Save frames to a video file
     * @param path The path to the video file
     * @param frames The frames to save
     * @param fps The number of frames per second
     */
    void saveFrames(const std::string& path, frames_ref& frames,
                    const int fps = 24);
} // namespace frame_helper
