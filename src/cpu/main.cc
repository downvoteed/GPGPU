#include <iostream>
#include <chrono>
#include <thread>
#include <functional>
#include <boost/program_options.hpp>

#include <frame-helper.hh>
#include <color-helper.hh>
#include <texture-helper.hh>
#include <segmentation-helper.hh>

#include "../stats/rss.hh"

namespace po = boost::program_options;

/**
 * Segment a frame based on the color and texture similarities with the background frame
 * @param i The index of the current frame
 * @param bg_features The texture features of the background frame
 * @param colored_frames The colored frames
 * @param gray_frames The grayscale frames
 * @param w The width of the frame
 * @param h The height of the frame
 * @param start The start time of the execution
 * @param verbose Whether to display the progress
 * @param size The number of frames
 * @param result The segmented frame
 */
void segment_frame(int i, texture_helper::feature_vector bg_features, cv::Mat colored_bg_frame, frame_helper::frames colored_frames, frame_helper::frames gray_frames, int w, int h, std::chrono::time_point<std::chrono::high_resolution_clock> start, bool verbose, int size, cv::Mat &result)
{
  auto end = std::chrono::high_resolution_clock::now();
  auto duration_avg = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  if (verbose)
  {
    // Display the progress
    const int progress = (int)(((float)i + 1) / (float)size * 100);
    std::cout << "Processing frame " << i + 1 << "/" << size << " (" << progress << "%)";
  }

  cv::Mat curr_colored_frame = colored_frames[i];
  cv::Mat curr_gray_frame = gray_frames[i];

  color_helper::similarity_vectors color_similarities = {
      color_helper::similarity_vector(w * h, 0),
      color_helper::similarity_vector(w * h, 0)};
  texture_helper::feature_vector features = {};

  for (int c = 0; c < w; c++)
  {
    for (int r = 0; r < h; r++)
    {
      // Compare the color components of the current frame with the background frame
      color_helper::similarity_vector color_similarity_vector = color_helper::compare(colored_bg_frame, curr_colored_frame, c, r);
      color_similarities[0][r * w + c] = color_similarity_vector[0];
      color_similarities[1][r * w + c] = color_similarity_vector[1];

      // Extract the texture features from the current frame
      features.push_back(texture_helper::calculateLBP(curr_gray_frame, c, r));
    }
  }

  // Compare the texture features of the current frame with the background frame
  texture_helper::similarity_vector texture_similarities = texture_helper::compare(bg_features, features);

  // Segment the current frame based on the color and texture similarities with the background frame
  segmentation_helper::layer_vector segments = segmentation_helper::segment(color_similarities, texture_similarities);
  cv::Mat segmented_frame = frame_helper::buildSegmentedFrame(segments, w, h);

  // Display the elapsed time
  if (verbose)
  {
    auto new_end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(new_end - end).count();
    duration_avg = (duration_avg + duration) / 2;
    auto estimated_remaining_time = duration_avg * (colored_frames.size() - i - 1) / 1000;

    std::cout << " - processed in " << duration << "ms - estimated time remaining: " << estimated_remaining_time << "s" << std::endl;
  }

  // Save the segmented frame
  result = segmented_frame;
}

int main(int argc, char **argv)
{
  LogRss logger;

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")("verbose", "enable verbose mode")("width", po::value<int>(), "set the width of the frame")("height", po::value<int>(), "set the height of the frame")("display", po::value<bool>()->default_value(true), "display the segmented frames")("output-path", po::value<std::string>(), "if set, save the video to the given path");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Check the number of parameters other than the flags
  if (argc < 2)
  {
    std::cout << "Usage: " << argv[0] << " <path-to-video>" << std::endl;
    return 1;
  }

  // Check if the help flag is set
  if (vm.count("help"))
  {
    std::cout << desc << "\n";
    return 1;
  }

  // Check if the verbose flag is set
  bool verbose = vm.count("verbose") > 0;

  // Start a timer to measure the execution time
  auto start = std::chrono::high_resolution_clock::now();

  // Read the frames from the dataset in color
  const std::optional<int> width = vm.count("width") > 0 ? std::make_optional(vm["width"].as<int>()) : std::nullopt;
  const std::optional<int> height = vm.count("height") > 0 ? std::make_optional(vm["height"].as<int>()) : std::nullopt;

  frame_helper::frames_vector frames_vector = frame_helper::readFrames(argv[1], width, height);
  frame_helper::frames colored_frames = frames_vector[0];
  frame_helper::frames gray_frames = frames_vector[1];

  // log RAM usage and display it
  if (verbose)
  {
    logger.begin_rss_loging();
  }

  if (colored_frames.size() == 0)
  {
    std::cout << "No colored frames found!" << std::endl;
    return 1;
  }

  if (gray_frames.size() == 0)
  {
    std::cout << "No grayscale frames found!" << std::endl;
    return 1;
  }

  if (verbose)
  {
    std::cout << colored_frames.size() << " frames loaded!";
  }

  // Extract the first frame from the dataset as the background
  cv::Mat colored_bg_frame = colored_frames[0];
  cv::Mat gray_bg_frame = gray_frames[0];
  const int w = colored_bg_frame.cols;
  const int h = colored_bg_frame.rows;

  if (verbose)
  {
    std::cout << " - Frame size: " << w << "x" << h << std::endl;
  }

  // Extract the texture features from the background frame in grayscale
  texture_helper::feature_vector bg_features = {};
  for (int c = 0; c < w; c++)
  {
    for (int r = 0; r < h; r++)
    {
      bg_features.push_back(texture_helper::calculateLBP(gray_bg_frame, c, r));
    }
  }

  // Process the frames
  std::vector<std::thread> threads;
  frame_helper::frames_ref segmented_frames = {};
  for (int i = 1; i < colored_frames.size(); i++)
  {
    cv::Mat* result = new cv::Mat();
    threads.push_back(std::thread(segment_frame, i, bg_features, colored_bg_frame, colored_frames, gray_frames, w, h, start, verbose, colored_frames.size(), std::ref(*result)));
    segmented_frames.push_back(result);
  }

  // Wait for all threads to finish
  for (std::thread &t : threads)
  {
    t.join();
  }

  auto end = std::chrono::high_resolution_clock::now();

  if (verbose)
  {
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Processing completed in " << total_duration << "ms" << std::endl;
  }

  // Save the segmented frames to a video file
  if (vm.count("output-path"))
  {
    std::string output_path = vm["output-path"].as<std::string>();
    frame_helper::saveFrames(output_path, segmented_frames);
  }

  // Display the segmented frames one by one and wait for a key press to display the next one
  if (vm["display"].as<bool>())
  {
    for (int i = 0; i < segmented_frames.size(); i++)
    {
      frame_helper::showFrame("Segmented Frame " + std::to_string(i) + "/" + std::to_string(segmented_frames.size()), *(segmented_frames[i]));
    }
  }

  return 0;
}
