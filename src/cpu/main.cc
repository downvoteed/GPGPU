#include <boost/asio/thread_pool.hpp>
#include <boost/program_options.hpp>
#include <iostream>

#include "../stats/rss.hh"
#include <logger.hh>
#include <video-processor.hh>
#include <webcam-processor.hh>

namespace po = boost::program_options;

int main(int argc, char **argv) {
  // LogRss logger;

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")("verbose,v",
                                                       "enable verbose mode")(
      "log-file,l", po::value<std::string>(), "set the log file path")(
      "log-level,L", po::value<std::string>()->default_value("info"),
      "set the logging level)")(
      "jobs,j",
      po::value<unsigned int>()->default_value(1)->implicit_value(
          std::thread::hardware_concurrency()),
      "set the number of threads to use")("input,i", po::value<std::string>(),
                                          "set the input video path")(
      "width", po::value<unsigned int>(), "set the width of the frame")(
      "height", po::value<unsigned int>(), "set the height of the frame")(
      "display,d", po::value<bool>()->default_value(true),
      "display the segmented frames")(
      "output,o", po::value<std::string>(),
      "if set, save the video to the given path")(
      "fps,f", po::value<unsigned int>()->default_value(24),
      "set the FPS of the video")("webcam,w", "use the webcam as input")(
      "background-optimizer",
      "use the background optimizer (is default for the webcam)");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Check if the help flag is set
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  // Exit if both the input and the webcam flags are set
  if (vm.count("input") && vm.count("webcam")) {
    std::cout << "The input and the webcam flags cannot be set at the same "
                 "time!\n";
    return 1;
  }

  // Exit if neither the input nor the webcam flags are set
  if (!vm.count("input") && !vm.count("webcam")) {
    std::cout << "Either the input or the webcam flag must be set!\n";
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

  const unsigned int max_threads = std::thread::hardware_concurrency();
  if (num_threads == 0 || num_threads > max_threads) {
    num_threads = max_threads;
  }

  if (verbose) {
    BOOST_LOG_TRIVIAL(info) << "Using " << num_threads << " threads";
  }

  // Create the thread pool
  boost::asio::thread_pool pool(num_threads);

  // If an input path is provided, process the video
  if (vm.count("input")) {
    // Determine the input path
    const std::string input_path = vm["input"].as<std::string>();

    // Determine the width and height of the frame
    const std::optional<unsigned int> width =
        vm.count("width") > 0
            ? std::make_optional(vm["width"].as<unsigned int>())
            : std::nullopt;
    const std::optional<unsigned int> height =
        vm.count("height") > 0
            ? std::make_optional(vm["height"].as<unsigned int>())
            : std::nullopt;

    // Determine the output path
    const std::optional<std::string> output_path =
        vm.count("output-path") > 0
            ? std::make_optional(vm["output-path"].as<std::string>())
            : std::nullopt;

    process_video(verbose, input_path, width, height, output_path,
                  vm["display"].as<bool>(), vm["fps"].as<unsigned int>(),
                  vm.count("background-optimizer") > 0, pool);
  }

  // If the webcam flag is set, process the webcam stream
  if (vm.count("webcam")) {
    process_webcam(verbose);
  }

  return 0;
}
