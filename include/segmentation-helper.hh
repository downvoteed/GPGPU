#pragma once

#include <color-helper.hh>
#include <logger.hh>
#include <texture-helper.hh>
#include <vector>

namespace segmentation_helper {
const float THRESHOLD = 0.67;
const float ALPHA = 0.1;

void bg_optimization(cv::Mat *colored_bg_frame, const cv::Mat &colored_frame,
                     const unsigned int w, const unsigned int i) {
  // Update the background model
  const unsigned int c = i % w;
  const unsigned int r = i / w;

  cv::Vec3b pixel = colored_frame.at<cv::Vec3b>(r, c);
  cv::Vec3b bg_pixel = colored_bg_frame->at<cv::Vec3b>(r, c);

  bg_pixel[0] = ALPHA * pixel[0] + (1 - ALPHA) * bg_pixel[0];
  bg_pixel[1] = ALPHA * pixel[1] + (1 - ALPHA) * bg_pixel[1];
  bg_pixel[2] = ALPHA * pixel[2] + (1 - ALPHA) * bg_pixel[2];

  colored_bg_frame->at<cv::Vec3b>(r, c) = bg_pixel;
}

/**
 * Segment a frame into foreground and background
 * @param color_similarities The color similarities between the current frame
 * and the background frame
 * @param bg_features The texture features of the background frame
 * @param features The texture features of the current frame
 * @return The values of the segments between 0 and 1, where 0 is the background
 */
void segment(const color_helper::similarity_vectors &color_similarities,
             const texture_helper::feature_vector &bg_features,
             const texture_helper::feature_vector &features,
             const unsigned int w, cv::Mat &result,
             const bool should_extract_bg, cv::Mat *colored_bg_frame,
             const cv::Mat &colored_frame) {
  // Calculate the weighted sum of the color and texture similarities
  for (unsigned long i = 0; i < color_similarities[0].size(); i++) {
    float r = color_similarities[0][i];
    float g = color_similarities[1][i];

    // Compare the texture features of the current frame with the background
    // frame
    float t = texture_helper::compare(i, bg_features, features);

    // Sort the similarities in ascending order
    float s1, s2, s3;
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
    const float similarity = s1 * 0.1 + s2 * 0.3 + s3 * 0.6;

    // If the similarity is greater than 0.67, it is the foreground
    const uint8_t value = similarity >= THRESHOLD ? 0 : 1;

    // Background model optimization
    if (should_extract_bg && value == 1) {
      bg_optimization(colored_bg_frame, colored_frame, w, i);
    }

    // Build the segmented frame
    frame_helper::buildSegmentedFrame(result, i, value, w);
  }
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
void segment_frame(const int i, const unsigned int size,
                   const texture_helper::feature_vector &bg_features,
                   cv::Mat *colored_bg_frame, const cv::Mat &colored_frame,
                   const cv::Mat &gray_frame, const unsigned int w,
                   const unsigned int h, const bool verbose, cv::Mat &result,
                   const bool should_extract_bg) {
  auto start = std::chrono::high_resolution_clock::now();

  // Display the progress
  if (verbose) {
    const int progress = (int)(((float)i) / (float)size * 100);

    BOOST_LOG_TRIVIAL(info)
        << "Processing frame: " << i << "/" << size << " (" << progress << "%)";
  }

  color_helper::similarity_vectors *color_similarities =
      new color_helper::similarity_vectors{
          color_helper::similarity_vector(w * h, 0),
          color_helper::similarity_vector(w * h, 0)};
  texture_helper::feature_vector *features =
      new texture_helper::feature_vector();

  for (unsigned int c = 0; c < w; c++) {
    for (unsigned int r = 0; r < h; r++) {
      float r_ratio = 0;
      float g_ratio = 0;

      // Compare the color components of the current frame with the
      // background frame
      color_helper::compare(*colored_bg_frame, colored_frame, c, r, r_ratio,
                            g_ratio);
      (*color_similarities)[0][r * w + c] = r_ratio;
      (*color_similarities)[1][r * w + c] = g_ratio;

      // Extract the texture features from the current frame
      features->push_back(texture_helper::calculateLBP(gray_frame, c, r));
    }
  }

  // Segment the current frame based on the color and texture similarities with
  // the background frame
  segmentation_helper::segment(*color_similarities, bg_features, *features, w,
                               result, should_extract_bg, colored_bg_frame,
                               colored_frame);

  // Log the duration of the segmentation
  if (verbose) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    BOOST_LOG_TRIVIAL(info)
        << "Frame " << i << "/" << size << " segmented in " << duration << "ms";
  }

  // Free the memory
  delete color_similarities;
  delete features;
}

} // namespace segmentation_helper
