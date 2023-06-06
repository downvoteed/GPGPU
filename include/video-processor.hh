#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <chrono>
#include <frame-helper.hh>
#include <segmentation-helper.hh>
#include <texture-helper.hh>

void process_video(const bool verbose, const std::string &video_path,
                   const std::optional<unsigned int> width,
                   const std::optional<unsigned int> height,
                   boost::asio::thread_pool &pool,
                   const std::optional<std::string> output_path,
                   const bool display, const unsigned int fps) {
  // Start a timer to measure the execution time
  const auto start = std::chrono::high_resolution_clock::now();

  if (verbose) {
    BOOST_LOG_TRIVIAL(info) << "Loading dataset";
  }

  const frame_helper::frames_vector *frames_vector =
      frame_helper::readFrames(video_path, width, height);
  const frame_helper::frames &colored_frames = frames_vector->at(0);
  const frame_helper::frames &gray_frames = frames_vector->at(1);

  // log RAM usage and display it
  if (verbose) {
    // logger.begin_rss_loging();
  }

  if (colored_frames.size() == 0) {
    std::cerr << "No colored frames found!" << std::endl;
    exit(1);
  }

  if (gray_frames.size() == 0) {
    std::cerr << "No grayscale frames found!" << std::endl;
    exit(1);
  }

  if (verbose) {
    BOOST_LOG_TRIVIAL(info) << "Dataset loaded";
    BOOST_LOG_TRIVIAL(info) << "- " << colored_frames.size() << " frames";
  }

  // Extract the first frame from the dataset as the background
  const cv::Mat &colored_bg_frame = colored_frames[0];
  const cv::Mat &gray_bg_frame = gray_frames[0];
  const unsigned int w = colored_bg_frame.cols;
  const unsigned int h = colored_bg_frame.rows;

  if (verbose) {
    BOOST_LOG_TRIVIAL(info) << "- Frame size: " << w << "x" << h;
    BOOST_LOG_TRIVIAL(info) << "Extracting the background frame";
  }

  // Extract the texture features from the background frame in grayscale
  texture_helper::feature_vector *bg_features =
      new texture_helper::feature_vector();
  for (unsigned int c = 0; c < w; c++) {
    for (unsigned int r = 0; r < h; r++) {
      bg_features->push_back(texture_helper::calculateLBP(gray_bg_frame, c, r));
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  if (verbose) {
    BOOST_LOG_TRIVIAL(info)
        << "Background frame extracted in "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
               .count()
        << "ms";
  }

  // Process the frames
  frame_helper::frames_ref segmented_frames = {};
  for (unsigned long i = 1; i < colored_frames.size(); i++) {
    // Add a task to the thread pool
    cv::Mat *result = new cv::Mat();

    boost::asio::post(pool, [i, bg_features, colored_bg_frame, colored_frames,
                             gray_frames, w, h, verbose, result] {
      segmentation_helper::segment_frame(
          i, colored_frames.size(), *bg_features, colored_bg_frame,
          colored_frames[i], gray_frames[i], w, h, verbose, std::ref(*result));

      // Free the memory by dropping the const
      const_cast<cv::Mat &>(colored_frames[i]).release();
      const_cast<cv::Mat &>(gray_frames[i]).release();
    });

    segmented_frames.push_back(result);
  }

  // Wait for all threads to finish
  pool.join();

  if (verbose) {
    end = std::chrono::high_resolution_clock::now();
    const auto total_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    BOOST_LOG_TRIVIAL(info)
        << "Processing completed in " << total_duration << "ms";
  }

  // Save the segmented frames to a video file
  if (output_path.has_value()) {
    if (verbose) {
      BOOST_LOG_TRIVIAL(info)
          << "Saving the segmented frames to " << output_path.value();
    }

    frame_helper::saveFrames(output_path.value(), segmented_frames, fps);

    if (verbose) {
      BOOST_LOG_TRIVIAL(info) << "Segmented frames saved";
    }
  }

  // Display the segmented frames one by one and wait for a key press to display
  // the next one
  if (display) {
    for (unsigned long i = 0; i < segmented_frames.size(); i++) {
      frame_helper::showFrame("Segmented Frame " + std::to_string(i) + "/" +
                                  std::to_string(segmented_frames.size()),
                              *(segmented_frames[i]));
    }
  }

  // Free the memory
  delete frames_vector;
  delete bg_features;
  for (unsigned long i = 0; i < segmented_frames.size(); i++) {
    segmented_frames[i]->release();
    delete segmented_frames[i];
  }
}