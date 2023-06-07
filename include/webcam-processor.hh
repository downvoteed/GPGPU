#include <chrono>
#include <logger.hh>
#include <opencv2/opencv.hpp>
#include <segmentation-helper.hh>
#include <texture-helper.hh>

void process_webcam(const bool verbose) {
  // Open the webcam
  cv::VideoCapture webcam(0);
  if (!webcam.isOpened()) {
    std::cerr << "Failed to open the webcam!" << std::endl;
    exit(1);
  }

  // Get the width and height of the webcam stream
  const unsigned int w = webcam.get(cv::CAP_PROP_FRAME_WIDTH);
  const unsigned int h = webcam.get(cv::CAP_PROP_FRAME_HEIGHT);

  if (verbose) {
    BOOST_LOG_TRIVIAL(info) << "Webcam stream opened";
    BOOST_LOG_TRIVIAL(info) << "- Frame size: " << w << "x" << h;
  }

  // Create the window to display the webcam stream
  cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);

  // Store the background frames and features
  cv::Mat *colored_bg_frame = nullptr;
  texture_helper::feature_vector *bg_features =
      new texture_helper::feature_vector();

  // Keep track of the average execution time
  auto total_duration = std::chrono::high_resolution_clock::duration::zero();
  unsigned int total_frames = 0;

  bool should_extract_bg = true;

  // Main loop
  while (true) {
    // Read a frame from the webcam
    cv::Mat frame;
    webcam >> frame;

    // Convert the frame to grayscale
    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

    // If ESC or Q is pressed, exit the loop
    const int key = cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q') {
      break;
    }

    // If no background features are available, extract them from the frame
    if (should_extract_bg) {
      if (verbose) {
        BOOST_LOG_TRIVIAL(info) << "Extracting the background frame";
      }

      // Set the background frames
      if (colored_bg_frame == nullptr) {
        colored_bg_frame = new cv::Mat(frame.clone());
      }

      cv::Mat gray_bg_frame;
      cv::cvtColor(*colored_bg_frame, gray_bg_frame, cv::COLOR_BGR2GRAY);

      // Extract the texture features from the background frame in grayscale
      for (unsigned int c = 0; c < w; c++) {
        for (unsigned int r = 0; r < h; r++) {
          bg_features->push_back(
              texture_helper::calculateLBP(gray_bg_frame, c, r));
        }
      }

      if (verbose) {
        BOOST_LOG_TRIVIAL(info) << "- Background features extracted";
      }

      // Release the gray background frame
      gray_bg_frame.release();
    }

    // Start a timer to measure the execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Segment the frame
    cv::Mat *result = new cv::Mat(h, w, CV_8UC1);
    segmentation_helper::segment_frame(0, 0, *bg_features, colored_bg_frame,
                                       frame, gray_frame, w, h, false,
                                       std::ref(*result));
    // Display the frame
    cv::imshow("Webcam", *result);

    if (verbose) {
      // Stop the timer
      auto stop = std::chrono::high_resolution_clock::now();

      // Calculate the elapsed time
      auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

      // Update the average execution time
      total_duration += duration;
      total_frames++;

      BOOST_LOG_TRIVIAL(info)
          << "Frame segmented in " << duration.count() << "ms ("
          << total_frames / std::chrono::duration_cast<std::chrono::seconds>(
                                total_duration)
                                .count()
          << "fps)";
    }

    // Free the matrix
    frame.release();
    gray_frame.release();

    result->release();
    delete result;
  }

  // Release the webcam
  webcam.release();

  // Release the window
  cv::destroyWindow("Webcam");

  // Release the colored background frame
  colored_bg_frame->release();

  // Free the memory
  delete bg_features;
  delete colored_bg_frame;
}