#pragma once

#include <color-helper.hh>
#include <logger.hh>
#include <texture-helper.hh>
#include <vector>

namespace segmentation_helper {
using layer_vector = std::vector<uint8_t>;

/**
 * Segment a frame into foreground and background
 * @param color_similarities The color similarities between the current frame
 * and the background frame
 * @param texture_similarities The texture similarities between the current
 * frame and the background frame
 * @return The a vector of boolean values (1 or 0) representing the foreground
 * or the background
 */
layer_vector segment(color_helper::similarity_vectors color_similarities,
                     texture_helper::similarity_vector texture_similarities) {
  layer_vector segments;

  // Define the factors for the color and texture similarities
  std::vector<double> factors = {0.1, 0.3, 0.6};

  // Calculate the weighted sum of the color and texture similarities
  for (unsigned long i = 0; i < color_similarities[0].size(); i++) {
    double r = color_similarities[0][i];
    double g = color_similarities[1][i];
    double t = texture_similarities[i];

    // Sort the similarities in ascending order
    std::vector<double> similarities = {r, g, t};
    std::sort(similarities.begin(), similarities.end());

    // Multiply the similarities with the factors
    for (unsigned long j = 0; j < similarities.size(); j++) {
      similarities[j] *= factors[j];
    }

    // Calculate the weighted sum of the similarities and threshold it
    double similarity = similarities[0] + similarities[1] + similarities[2];

    // If the similarity is greater than 0.67, it is the foreground
    if (similarity >= 0.67) {
      segments.push_back(0);
    }
    // Otherwise, it is the background
    else {
      segments.push_back(1);
    }
  }

  return segments;
}

/**
 * Segment a frame based on the color and texture similarities with the
 * background frame
 * @param i The index of the current frame
 * @param bg_features The texture features of the background frame
 * @param colored_bg_frame The background frame in color
 * @param colored_frames The colored frames
 * @param gray_frames The grayscale frames
 * @param w The width of the frame
 * @param h The height of the frame
 * @param verbose Whether to display the progress
 * @param result The segmented frame
 */
void segment_frame(int i, texture_helper::feature_vector bg_features,
                   cv::Mat colored_bg_frame,
                   frame_helper::frames colored_frames,
                   frame_helper::frames gray_frames, int w, int h, bool verbose,
                   cv::Mat &result) {
  auto start = std::chrono::high_resolution_clock::now();

  // Display the progress
  if (verbose) {
    const int progress = (int)(((float)i) / (float)colored_frames.size() * 100);

    BOOST_LOG_TRIVIAL(info)
        << "Processing frame: " << i << "/" << colored_frames.size() << " ("
        << progress << "%)";
  }

  cv::Mat curr_colored_frame = colored_frames[i];
  cv::Mat curr_gray_frame = gray_frames[i];

  color_helper::similarity_vectors color_similarities = {
      color_helper::similarity_vector(w * h, 0),
      color_helper::similarity_vector(w * h, 0)};
  texture_helper::feature_vector features = {};

  for (int c = 0; c < w; c++) {
    for (int r = 0; r < h; r++) {
      // Compare the color components of the current frame with the background
      // frame
      color_helper::similarity_vector color_similarity_vector =
          color_helper::compare(colored_bg_frame, curr_colored_frame, c, r);
      color_similarities[0][r * w + c] = color_similarity_vector[0];
      color_similarities[1][r * w + c] = color_similarity_vector[1];

      // Extract the texture features from the current frame
      features.push_back(texture_helper::calculateLBP(curr_gray_frame, c, r));
    }
  }

  // Compare the texture features of the current frame with the background frame
  texture_helper::similarity_vector texture_similarities =
      texture_helper::compare(bg_features, features);

  // Segment the current frame based on the color and texture similarities with
  // the background frame
  segmentation_helper::layer_vector segments =
      segmentation_helper::segment(color_similarities, texture_similarities);
  cv::Mat segmented_frame = frame_helper::buildSegmentedFrame(segments, w, h);

  if (verbose) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    BOOST_LOG_TRIVIAL(info) << "Frame " << i << "/" << colored_frames.size()
                            << " segmented in " << duration << "ms";
  }

  // Save the segmented frame
  result = segmented_frame;
}

} // namespace segmentation_helper
