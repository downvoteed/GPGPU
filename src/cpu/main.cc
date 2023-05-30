#include <iostream>
#include <chrono>

#include <frame-helper.hh>
#include <color-helper.hh>
#include <texture-helper.hh>
#include <segmentation-helper.hh>

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    std::cout << "Usage: " << argv[0] << " <path-to-video>" << std::endl;
    return 1;
  }

  // Start a timer to measure the execution time
  auto start = std::chrono::high_resolution_clock::now();

  // Read the frames from the dataset in color
  std::cout << "Loading colored frames..." << std::endl;
  frame_helper::frames colored_frames = frame_helper::readFrames(argv[1]);
  if (colored_frames.size() == 0)
  {
    std::cout << "No colored frames found!" << std::endl;
    return 1;
  }
  std::cout << colored_frames.size() << " colored frames loaded!" << std::endl;

  // Read the frames from the dataset in grayscale
  std::cout << "Loading gray frames..." << std::endl;
  frame_helper::frames gray_frames = frame_helper::readFrames(argv[1], true);
  if (gray_frames.size() == 0)
  {
    std::cout << "No gray frames found!" << std::endl;
    return 1;
  }
  std::cout << gray_frames.size() << " gray frames loaded!" << std::endl;

  // Extract the first frame from the dataset as the background
  cv::Mat colored_bg_frame = colored_frames[0];
  cv::Mat gray_bg_frame = gray_frames[0];

  // Process the frames
  frame_helper::frames segmented_frames;
  for (int i = 1; i < colored_frames.size(); i++)
  {
    std::cout << "Processing frame " << i << "..." << std::endl;

    cv::Mat curr_colored_frame = colored_frames[i];
    cv::Mat curr_gray_frame = gray_frames[i];

    // Get the color components (R, G) from the current frame and the background frame
    color_helper::color_components components1 = color_helper::convert(colored_bg_frame);
    color_helper::color_components components2 = color_helper::convert(curr_colored_frame);

    // Compare the color components of the current frame with the background frame
    color_helper::similarity_vectors color_similarities = color_helper::compare(components1, components2);

    // Extract the texture features from the current frame and the background frame in grayscale
    texture_helper::feature_vector features1 = texture_helper::extract(gray_bg_frame);
    texture_helper::feature_vector features2 = texture_helper::extract(curr_gray_frame);

    // Compare the texture features of the current frame with the background frame
    texture_helper::similarity_vector texture_similarities = texture_helper::compare(features1, features2);

    // Segment the current frame based on the color and texture similarities with the background frame
    segmentation_helper::layer_vector segments = segmentation_helper::segment(color_similarities, texture_similarities);
    cv::Mat segmented_frame = frame_helper::buildSegmentedFrame(segments, colored_bg_frame.cols, colored_bg_frame.rows);

    // Add the segmented frame to the list of segmented frames
    segmented_frames.push_back(segmented_frame);

    // Display the elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time elapsed: " << duration << "ms" << std::endl;
  }

  // Display the segmented frames one by one and wait for a key press to display the next one
  for (int i = 0; i < segmented_frames.size(); i++)
  {
    frame_helper::showFrame("Segmented Frame", segmented_frames[i]);
  }

  return 0;
}
