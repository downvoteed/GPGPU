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

  // Store the background colored frame and features
  cv::Mat colored_bg_frame = cv::Mat();
  texture_helper::feature_vector *bg_features =
      new texture_helper::feature_vector();

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
    if (bg_features->size() == 0) {
      if (verbose) {
        BOOST_LOG_TRIVIAL(info) << "Extracting the background frame";
      }

      // Extract the texture features from the background frame in grayscale
      for (unsigned int c = 0; c < w; c++) {
        for (unsigned int r = 0; r < h; r++) {
          bg_features->push_back(
              texture_helper::calculateLBP(gray_frame, c, r));
        }
      }

      if (verbose) {
        BOOST_LOG_TRIVIAL(info) << "- Background features extracted";
      }

      // Set the colored background frame
      colored_bg_frame = frame.clone();

      // Free the matrix
      gray_frame.release();
      frame.release();

      continue;
    }

    // Add a task to the thread pool
    cv::Mat *result = new cv::Mat();

    segmentation_helper::segment_frame(0, 0, *bg_features, colored_bg_frame,
                                       frame, gray_frame, w, h, false,
                                       std::ref(*result));

    // Display the frame
    cv::imshow("Webcam", *result);

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

  // Free the memory
  delete bg_features;
}