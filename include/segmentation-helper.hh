#pragma once

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <color-helper.hh>
#include <logger.hh>
#include <texture-helper.hh>
#include <vector>

namespace segmentation_helper {
const float THRESHOLD = 0.67;

/**
 * Optimize the background model
 * @param colored_bg_frame The background frame in color
 * @param colored_frame The current frame in color
 * @param w The width of the frame
 * @param i The index of the current pixel
 * @param alpha The alpha value for the background optimizer
 */
void bg_optimization(cv::Mat *colored_bg_frame, const cv::Mat &colored_frame,
                     const unsigned int w, const unsigned int i,
                     const double alpha) {
  // Update the background model
  const unsigned int c = i % w;
  const unsigned int r = i / w;

  cv::Vec3b pixel = colored_frame.at<cv::Vec3b>(r, c);
  cv::Vec3b bg_pixel = colored_bg_frame->at<cv::Vec3b>(r, c);

  bg_pixel[0] = alpha * pixel[0] + (1 - alpha) * bg_pixel[0];
  bg_pixel[1] = alpha * pixel[1] + (1 - alpha) * bg_pixel[1];
  bg_pixel[2] = alpha * pixel[2] + (1 - alpha) * bg_pixel[2];

  colored_bg_frame->at<cv::Vec3b>(r, c) = bg_pixel;
}

/**
 * Segment a frame into foreground and background
 * @param color_similarities The color similarities between the current frame
 * and the background frame
 * @param bg_features The texture features of the background frame
 * @param features The texture features of the current frame
 * @param w The width of the frame
 * @param result The segmented frame
 * @param colored_bg_frame The background frame in color
 * @param colored_frame The current frame in color
 * @param alpha The alpha value for the background optimizer
 */
void segment(const color_helper::similarity_vectors &color_similarities,
             const texture_helper::feature_vector &bg_features,
             const texture_helper::feature_vector &features,
             const unsigned int w, cv::Mat &result, cv::Mat *colored_bg_frame,
             const cv::Mat &colored_frame, const double alpha) {
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
    if (alpha > 0 && value == 1) {
      bg_optimization(colored_bg_frame, colored_frame, w, i, alpha);
    }

    // Build the segmented frame
    frame_helper::buildSegmentedFrame(result, i, value, w);
  }
}

/**
 * Segment a block of a frame based on the color and texture similarities with
 * the background frame
 * @param min_c The minimum c coordinate of the block
 * @param max_c The maximum c coordinate of the block
 * @param min_r The minimum r coordinate of the block
 * @param max_r The maximum r coordinate of the block
 * @param color_similarities The color similarities between the current frame
 * and the background frame
 * @param features The texture features of the current frame
 * @param colored_bg_frame The background frame in color
 * @param colored_frame The current frame in color
 * @param gray_frame The current frame in grayscale
 * @param w The width of the frame
*/
void segment_block(const unsigned int min_c, const unsigned int max_c,
                   const unsigned int min_r, const unsigned int max_r,
                   color_helper::similarity_vectors *color_similarities,
                   texture_helper::feature_vector *features,
                   cv::Mat *colored_bg_frame, const cv::Mat &colored_frame,
                   const cv::Mat &gray_frame, const unsigned int w) {
  for (unsigned int c = min_c; c < max_c; c++) {
    for (unsigned int r = min_r; r < max_r; r++) {
      float r_ratio = 0;
      float g_ratio = 0;

      // Compare the color components of the current frame with the
      // background frame
      color_helper::compare(*colored_bg_frame, colored_frame, c, r, r_ratio,
                            g_ratio);
      (*color_similarities)[0][r * w + c] = r_ratio;
      (*color_similarities)[1][r * w + c] = g_ratio;

      // Extract the texture features from the current frame
      features->at(r * w + c) = texture_helper::calculateLBP(gray_frame, c, r);
    }
  }
}

/**
 * Segment a frame based on the color and texture similarities with the
 * background frame
 * @param i The index of the current frame
 * @param size The total number of frames
 * @param bg_features The texture features of the background frame
 * @param colored_bg_frame The background frame in color
 * @param colored_frame The current frame in color
 * @param gray_frame The current frame in grayscale
 * @param w The width of the frame
 * @param h The height of the frame
 * @param verbose Whether to display the progress
 * @param result The segmented frame
 * @param num_threads The number of threads to use
 * @param alpha The alpha value for the background optimizer
 */
void segment_frame(const int i, const unsigned int size,
                   const texture_helper::feature_vector &bg_features,
                   cv::Mat *colored_bg_frame, const cv::Mat &colored_frame,
                   const cv::Mat &gray_frame, const unsigned int w,
                   const unsigned int h, const bool verbose, cv::Mat &result,
                   const unsigned int num_threads, const double alpha) {
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
      new texture_helper::feature_vector(w * h, 0);

  // Closest power of 2 to the number of threads in the pool
  const unsigned int block_size = h / num_threads;

  boost::asio::thread_pool pool(num_threads);

  // Segment the frame in blocks
  for (unsigned int j = 0; j < num_threads; j++) {
    const unsigned int min_r = j * block_size;
    const unsigned int max_r = std::min((j + 1) * block_size, h);
    boost::asio::post(pool,
                      std::bind(segment_block, 0, w, min_r, max_r,
                                color_similarities, features, colored_bg_frame,
                                colored_frame, std::ref(gray_frame), w));
  }

  // Wait for all threads to finish
  pool.join();

  // Segment the current frame based on the color and texture similarities with
  // the background frame
  segmentation_helper::segment(*color_similarities, bg_features, *features, w,
                               result, colored_bg_frame, colored_frame, alpha);

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
