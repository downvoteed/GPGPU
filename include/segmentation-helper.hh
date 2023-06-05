#pragma once

#include <color-helper.hh>
#include <logger.hh>
#include <texture-helper.hh>
#include <vector>

namespace segmentation_helper {
const double THRESHOLD = 0.67;

/**
 * Segment a frame into foreground and background
 * @param color_similarities The color similarities between the current frame
 * and the background frame
 * @param bg_features The texture features of the background frame
 * @param features The texture features of the current frame
 * @return The values of the segments between 0 and 1, where 0 is the background
 */
const cv::Mat &
segment(const color_helper::similarity_vectors &color_similarities,
        const texture_helper::feature_vector &bg_features,
        const texture_helper::feature_vector &features, const unsigned int w,
        const unsigned int h) {
  cv::Mat *frame = new cv::Mat(h, w, CV_8UC1);

  // Calculate the weighted sum of the color and texture similarities
  for (unsigned long i = 0; i < color_similarities[0].size(); i++) {
    double r = color_similarities[0][i];
    double g = color_similarities[1][i];

    // Compare the texture features of the current frame with the background
    // frame
    uint8_t t = texture_helper::compare(i, bg_features, features);

    // Sort the similarities in ascending order
    double s1, s2, s3;
    if (r <= g && r <= t) {
      s1 = r;
      if (g <= t) {
        s2 = g;
        s3 = t;
      } else {
        s2 = t;
        s3 = g;
      }
    } else if (g <= r && g <= t) {
      s1 = g;
      if (r <= t) {
        s2 = r;
        s3 = t;
      } else {
        s2 = t;
        s3 = r;
      }
    } else {
      s1 = t;
      if (r <= g) {
        s2 = r;
        s3 = g;
      } else {
        s2 = g;
        s3 = r;
      }
    }

    // Calculate the weighted sum of the similarities and threshold it
    double similarity = s1 * 0.1 + s2 * 0.3 + s3 * 0.6;

    // If the similarity is greater than 0.67, it is the foreground
    frame_helper::buildSegmentedFrame(*frame, i,
                                      similarity >= THRESHOLD ? 0 : 1, w);
  }

  return *frame;
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
void segment_frame(const int i,
                   const texture_helper::feature_vector &bg_features,
                   const cv::Mat &colored_bg_frame,
                   const frame_helper::frames &colored_frames,
                   const frame_helper::frames &gray_frames,
                   const unsigned int w, const unsigned int h,
                   const bool verbose, cv::Mat &result) {
  auto start = std::chrono::high_resolution_clock::now();

  // Display the progress
  if (verbose) {
    const int progress = (int)(((float)i) / (float)colored_frames.size() * 100);

    BOOST_LOG_TRIVIAL(info)
        << "Processing frame: " << i << "/" << colored_frames.size() << " ("
        << progress << "%)";
  }

  const cv::Mat &curr_colored_frame = colored_frames[i];
  const cv::Mat &curr_gray_frame = gray_frames[i];

  color_helper::similarity_vectors *color_similarities =
      new color_helper::similarity_vectors{
          color_helper::similarity_vector(w * h, 0),
          color_helper::similarity_vector(w * h, 0)};
  texture_helper::feature_vector *features =
      new texture_helper::feature_vector();

  for (int c = 0; c < w; c++) {
    for (int r = 0; r < h; r++) {
      // Compare the color components of the current frame with the
      // background frame
      const color_helper::similarity_vector *color_similarity_vector =
          color_helper::compare(colored_bg_frame, curr_colored_frame, c, r);
      (*color_similarities)[0][r * w + c] = color_similarity_vector->at(0);
      (*color_similarities)[1][r * w + c] = color_similarity_vector->at(1);

      // Extract the texture features from the current frame
      features->push_back(texture_helper::calculateLBP(curr_gray_frame, c, r));

      // Free the memory
      delete color_similarity_vector;
    }
  }

  // Segment the current frame based on the color and texture similarities with
  // the background frame
  const cv::Mat &segmented_frame = segmentation_helper::segment(
      *color_similarities, bg_features, *features, w, h);

  // Log the duration of the segmentation
  if (verbose) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    BOOST_LOG_TRIVIAL(info) << "Frame " << i << "/" << colored_frames.size()
                            << " segmented in " << duration << "ms";
  }

  // Free the memory
  delete color_similarities;
  delete features;

  // Save the segmented frame
  result = segmented_frame;
}

} // namespace segmentation_helper
