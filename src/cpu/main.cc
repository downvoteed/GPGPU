#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>

#include "../stats/rss.hh"
#include <color-helper.hh>
#include <frame-helper.hh>
#include <logger.hh>
#include <segmentation-helper.hh>
#include <texture-helper.hh>

namespace po = boost::program_options;

int main(int argc, char **argv) {
  // LogRss logger;

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")("verbose,v",
                                                       "enable verbose mode")(
      "log-file,l", po::value<std::string>(), "set the log file path")(
      "width", po::value<unsigned int>(), "set the width of the frame")(
      "height", po::value<unsigned int>(), "set the height of the frame")(
      "display,d", po::value<bool>()->default_value(true),
      "display the segmented frames")(
      "output-path,o", po::value<std::string>(),
      "if set, save the video to the given path")(
      "jobs,j",
      po::value<unsigned int>()->default_value(1)->implicit_value(
          std::thread::hardware_concurrency()),
      "set the number of threads to use")(
      "fps,f", po::value<unsigned int>()->default_value(24),
      "set the FPS of the video")(
      "log-level,L", po::value<std::string>()->default_value("info"),
      "set the logging level)");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Check the number of parameters other than the flags
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <path-to-video>" << std::endl;
    return 1;
  }

  // Check if the help flag is set
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  // Set the logging level
  std::string log_level = vm["log-level"].as<std::string>();
  if (log_level != "trace" && log_level != "debug" && log_level != "info" &&
      log_level != "warning" && log_level != "error" && log_level != "fatal") {
    BOOST_LOG_TRIVIAL(error) << "Invalid logging level!";
    BOOST_LOG_TRIVIAL(error)
        << "Valid values are: trace, debug, info, warning, error, fatal";
    return 1;
  }

  set_logging_level(log_level);

  // Check if the verbose flag is set
  const bool verbose = vm.count("verbose") > 0 || vm.count("log-file") > 0;
  if (verbose) {
    auto log_file = vm.count("log-file") > 0
                        ? std::make_optional(vm["log-file"].as<std::string>())
                        : std::nullopt;

    init_logging(vm.count("verbose") > 0, log_file);
    BOOST_LOG_TRIVIAL(info) << "Starting the program";
  }

  // Determine the number of threads to use
  unsigned int num_threads = vm["jobs"].as<unsigned int>();
  if (num_threads < 0) {
    BOOST_LOG_TRIVIAL(error) << "Invalid number of threads!";
    return 1;
  }

  const unsigned int max_threads = std::thread::hardware_concurrency();
  if (num_threads == 0 || num_threads > max_threads) {
    num_threads = max_threads;
  }

  if (verbose) {
    BOOST_LOG_TRIVIAL(info) << "Using " << num_threads << " threads";
  }

  // Start a timer to measure the execution time
  const auto start = std::chrono::high_resolution_clock::now();

  // Read the frames from the dataset in color
  const std::optional<unsigned int> width =
      vm.count("width") > 0 ? std::make_optional(vm["width"].as<unsigned int>())
                            : std::nullopt;
  const std::optional<unsigned int> height =
      vm.count("height") > 0
          ? std::make_optional(vm["height"].as<unsigned int>())
          : std::nullopt;

  if (verbose) {
    BOOST_LOG_TRIVIAL(info) << "Loading dataset";
  }

  const frame_helper::frames_vector *frames_vector =
      frame_helper::readFrames(argv[1], width, height);
  const frame_helper::frames &colored_frames = frames_vector->at(0);
  const frame_helper::frames &gray_frames = frames_vector->at(1);

  // log RAM usage and display it
  if (verbose) {
    // logger.begin_rss_loging();
  }

  if (colored_frames.size() == 0) {
    std::cerr << "No colored frames found!" << std::endl;
    return 1;
  }

  if (gray_frames.size() == 0) {
    std::cerr << "No grayscale frames found!" << std::endl;
    return 1;
  }

  if (verbose) {
    BOOST_LOG_TRIVIAL(info) << "Dataset loaded";
    BOOST_LOG_TRIVIAL(info) << "- " << colored_frames.size() << " frames";
  }

  // Extract the first frame from the dataset as the background
  const cv::Mat &colored_bg_frame = colored_frames[0];
  const cv::Mat &gray_bg_frame = gray_frames[0];
  const int w = colored_bg_frame.cols;
  const int h = colored_bg_frame.rows;

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

  // Create a thread pool
  boost::asio::thread_pool pool(num_threads);

  // Process the frames
  frame_helper::frames_ref segmented_frames = {};
  for (unsigned long i = 1; i < colored_frames.size(); i++) {
    // Add a task to the thread pool
    cv::Mat *result = new cv::Mat();

    boost::asio::post(pool, [i, bg_features, colored_bg_frame, colored_frames,
                             gray_frames, w, h, verbose, result] {
      segmentation_helper::segment_frame(i, *bg_features, colored_bg_frame,
                                         colored_frames, gray_frames, w, h,
                                         verbose, std::ref(*result));
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
  if (vm.count("output-path")) {
    std::string output_path = vm["output-path"].as<std::string>();
    const unsigned int fps = vm["fps"].as<unsigned int>();

    if (verbose) {
      BOOST_LOG_TRIVIAL(info)
          << "Saving the segmented frames to " << output_path;
    }

    frame_helper::saveFrames(output_path, segmented_frames, fps);

    if (verbose) {
      BOOST_LOG_TRIVIAL(info) << "Segmented frames saved";
    }
  }

  // Display the segmented frames one by one and wait for a key press to display
  // the next one
  if (vm["display"].as<bool>()) {
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
    delete segmented_frames[i];
  }

  return 0;
}
