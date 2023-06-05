#pragma once

#include <logger.hh>
#include <opencv2/opencv.hpp>

namespace frame_helper {
using frames = std::vector<cv::Mat>;
using frames_ref = std::vector<cv::Mat *>;
using frames_vector = std::vector<frames>;

/**
 * Read a frame from a file
 * @param path The path to the file
 * @param gray Whether to read the frame in grayscale
 * @return The OpenCV frame
 */
const cv::Mat &readFrame(const std::string &path, const bool gray = false) {
  int flag = cv::IMREAD_COLOR;
  if (gray) {
    flag = cv::IMREAD_GRAYSCALE;
  }

  cv::Mat *frame = new cv::Mat();
  *frame = cv::imread(path, flag);

  if (frame->empty()) {
    BOOST_LOG_TRIVIAL(error) << "Could not read the image: " << path;
    exit(1);
  }

  return *frame;
}

/**
 * Show a frame in a window
 * @param name The name of the window
 * @param frame The OpenCV frame
 */
void showFrame(const std::string &name, const cv::Mat &frame) {
  cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
  cv::imshow(name, frame);
  cv::waitKey(0);

  cv::destroyWindow(name);
}

/**
 * Build a segmented frame from a vector of boolean values
 * (foreground/background)
 * @param value 0 for foreground, 1 for background
 * @param width The width of the frame
 * @return The segmented frame
 */
const cv::Mat &buildSegmentedFrame(cv::Mat &frame, const unsigned int i,
                                   const uint8_t value, const int width) {
  int c = i % width;
  int r = i / width;
  frame.at<uint8_t>(r, c) = value * 255;

  return frame;
}

/**
 * Read a video file and return its frames
 * @param path The path to the video file
 * @param width The width of the frames (optional)
 * @param height The height of the frames (optional)
 * @return The frames
 */
const frames_vector *
readFrames(const std::string &path,
           const std::optional<int> &width = std::nullopt,
           const std::optional<int> &height = std::nullopt) {
  cv::VideoCapture video(path);
  if (!video.isOpened()) {
    BOOST_LOG_TRIVIAL(error) << "Could not open the video: " << path;
    exit(1);
  }

  frames colored_frames = {};
  frames gray_frames = {};

  // Read the video frame by frame
  while (true) {
    cv::Mat frame;
    video >> frame;

    // Break the loop at the end of the video
    if (frame.empty()) {
      break;
    }

    // Resize the frame if needed
    if (width.has_value() || height.has_value()) {
      int w = width.value_or(frame.cols);
      int h = height.value_or(frame.rows);
      cv::resize(frame, frame, cv::Size(w, h));
    }

    // Add the frame to the frames vector
    colored_frames.push_back(frame);

    // Convert the frame to grayscale and add it to the frames vector
    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    gray_frames.push_back(gray_frame);
  }

  // Release the video capture object
  video.release();

  // Return the frames
  frames_vector *frames = new frames_vector();
  frames->push_back(colored_frames);
  frames->push_back(gray_frames);
  return frames;
}

/**
 * Save frames to a video file
 * @param path The path to the video file
 * @param frames The frames to save
 * @param fps The number of frames per second
 */
void saveFrames(const std::string &path, frames_ref &frames,
                const int fps = 24) {
  cv::Size frameSize(frames[0]->cols, frames[0]->rows);
  cv::VideoWriter video(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                        frameSize);
  if (!video.isOpened()) {
    BOOST_LOG_TRIVIAL(error) << "Could not open the video: " << path;
    exit(1);
  }

  // Write the frames to the video
  for (unsigned long i = 0; i < frames.size(); i++) {
    auto frame = frames[i];
    cv::Mat bgrFrame;

    // Check if the frame is grayscale (single channel)
    if (frames[i]->channels() == 1) {
      // Convert grayscale to BGR
      cv::cvtColor(*frame, bgrFrame, cv::COLOR_GRAY2BGR);
    } else {
      bgrFrame = *(frames[i]);
    }

    video.write(bgrFrame);
  }

  // Release the video writer object
  video.release();
}

} // namespace frame_helper
