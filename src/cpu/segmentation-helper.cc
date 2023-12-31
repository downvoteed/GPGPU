#include "segmentation-helper.hh"

#include "frame-helper.hh"

namespace segmentation_helper
{
    /**
     * Optimize the background model
     * @param colored_bg_frame The background frame in color
     * @param colored_frame The current frame in color
     * @param w The width of the frame
     * @param i The index of the current pixel
     * @param value The value of the current pixel
     * @param learning_rate The learning rate for adaptive alpha adjustment
     * @param weights The weights for the weighted average blending
     */
    void bg_optimization(cv::Mat* colored_bg_frame,
                         const cv::Mat& colored_frame, const unsigned int w,
                         const unsigned int i, const uint8_t value,
                         const double learning_rate, cv::Mat& weights)
    {
        // Update the background model
        const unsigned int c = i % w;
        const unsigned int r = i / w;

        cv::Vec3b pixel = colored_frame.at<cv::Vec3b>(r, c);
        cv::Vec3b bg_pixel = colored_bg_frame->at<cv::Vec3b>(r, c);

        double weight = weights.at<double>(r, c);
        double updated_weight = weight;

        if (value == 1)
        {
            // Compute the weighted average of the pixel values
            bg_pixel[0] = (1 - weight) * bg_pixel[0] + weight * pixel[0];
            bg_pixel[1] = (1 - weight) * bg_pixel[1] + weight * pixel[1];
            bg_pixel[2] = (1 - weight) * bg_pixel[2] + weight * pixel[2];

            // Update the weights with an adaptive learning rate
            updated_weight = learning_rate * weight + (1 - learning_rate);
        }

        // Update the weights
        weights.at<double>(r, c) = updated_weight;

        // Update the background frame with the optimized pixel value
        colored_bg_frame->at<cv::Vec3b>(r, c) = bg_pixel;
    }

    /**
     * Segment a frame into foreground and background
     * @param color_similarities The color similarities between the current
     * frame and the background frame
     * @param bg_features The texture features of the background frame
     * @param features The texture features of the current frame
     * @param w The width of the frame
     * @param result The segmented frame
     * @param colored_bg_frame The background frame in color
     * @param colored_frame The current frame in color
     * @param learning_rate The learning_rate value for the background optimizer
     */
    void segment(const color_helper::similarity_vectors& color_similarities,
                 const texture_helper::feature_vector& bg_features,
                 const texture_helper::feature_vector& features,
                 const unsigned int w, cv::Mat& result,
                 cv::Mat* colored_bg_frame, const cv::Mat& colored_frame,
                 const double learning_rate, cv::Mat& weights)
    {
        // Calculate the weighted sum of the color and texture similarities
        for (unsigned long i = 0; i < color_similarities[0].size(); i++)
        {
            float r = color_similarities[0][i];
            float g = color_similarities[1][i];

            // Compare the texture features of the current frame with the
            // background frame
            float t = texture_helper::compare(i, bg_features, features);

            // Sort the similarities in ascending order
            float s1, s2, s3;
            if (r <= g && r <= t)
            {
                s1 = r;
                if (g <= t)
                {
                    s2 = g;
                    s3 = t;
                }
                else
                {
                    s2 = t;
                    s3 = g;
                }
            }
            else if (g <= r && g <= t)
            {
                s1 = g;
                if (r <= t)
                {
                    s2 = r;
                    s3 = t;
                }
                else
                {
                    s2 = t;
                    s3 = r;
                }
            }
            else
            {
                s1 = t;
                if (r <= g)
                {
                    s2 = r;
                    s3 = g;
                }
                else
                {
                    s2 = g;
                    s3 = r;
                }
            }

            // Calculate the weighted sum of the similarities and threshold it
            const float similarity = s1 * 0.1 + s2 * 0.3 + s3 * 0.6;

            // If the similarity is greater than 0.67, it is the foreground
            const uint8_t value = similarity >= THRESHOLD ? 0 : 1;

            // Background model optimization
            if (learning_rate > 0)
            {
                bg_optimization(colored_bg_frame, colored_frame, w, i, value,
                                learning_rate, weights);
            }

            // Build the segmented frame
            frame_helper::buildSegmentedFrame(result, i, value, w);
        }
    }

    /**
     * Segment a block of a frame based on the color and texture similarities
     * with the background frame
     * @param min_c The minimum c coordinate of the block
     * @param max_c The maximum c coordinate of the block
     * @param min_r The minimum r coordinate of the block
     * @param max_r The maximum r coordinate of the block
     * @param color_similarities The color similarities between the current
     * frame and the background frame
     * @param features The texture features of the current frame
     * @param bg_colors The background color components
     * @param colored_frame The current frame in color
     * @param gray_frame The current frame in grayscale
     * @param w The width of the frame
     */
    void segment_block(const unsigned int min_c, const unsigned int max_c,
                       const unsigned int min_r, const unsigned int max_r,
                       color_helper::similarity_vectors* color_similarities,
                       texture_helper::feature_vector* features,
                       const color_helper::color_vectors& bg_colors,
                       const cv::Mat& colored_frame, const cv::Mat& gray_frame,
                       const unsigned int w)
    {
        for (unsigned int c = min_c; c < max_c; c++)
        {
            for (unsigned int r = min_r; r < max_r; r++)
            {
                float r_ratio = 0;
                float g_ratio = 0;

                // Compare the color components of the current frame with the
                // background frame
                color_helper::compare(bg_colors, colored_frame, c, r, r_ratio,
                                      g_ratio);
                (*color_similarities)[0][r * w + c] = r_ratio;
                (*color_similarities)[1][r * w + c] = g_ratio;

                // Extract the texture features from the current frame
                features->at(r * w + c) =
                    texture_helper::calculateLBP(gray_frame, c, r);
            }
        }
    }

    /**
     * Segment a frame based on the color and texture similarities with the
     * background frame
     * @param i The index of the current frame
     * @param size The total number of frames
     * @param colored_bg_frame The background frame in color
     * @param bg_features The texture features of the background frame
     * @param bg_colors The background color components
     * @param colored_frame The current frame in color
     * @param gray_frame The current frame in grayscale
     * @param w The width of the frame
     * @param h The height of the frame
     * @param verbose Whether to display the progress
     * @param result The segmented frame
     * @param num_threads The number of threads to use
     * @param learning_rate The learning_rate value for the background optimizer
     * @param weights The weights for the weighted average blending
     */
    void segment_frame(const int i, const unsigned int size,
                       cv::Mat* colored_bg_frame,
                       const texture_helper::feature_vector& bg_features,
                       const color_helper::color_vectors& bg_colors,
                       const cv::Mat& colored_frame, const cv::Mat& gray_frame,
                       const unsigned int w, const unsigned int h,
                       const bool verbose, cv::Mat& result,
                       const unsigned int num_threads,
                       const double learning_rate, cv::Mat& weights)
    {
        auto start = std::chrono::high_resolution_clock::now();

        // Display the progress
        if (verbose)
        {
            const int progress = (int)(((float)i) / (float)size * 100);

            BOOST_LOG_TRIVIAL(info) << "Processing frame: " << i << "/" << size
                                    << " (" << progress << "%)";
        }

        color_helper::similarity_vectors* color_similarities =
            new color_helper::similarity_vectors{
                color_helper::similarity_vector(w * h, 0),
                color_helper::similarity_vector(w * h, 0)
            };
        texture_helper::feature_vector* features =
            new texture_helper::feature_vector(w * h, 0);

        // Calculate the block size
        const unsigned int block_size = h / num_threads;

        boost::asio::thread_pool pool(num_threads);

        // Segment the frame in blocks
        for (unsigned int j = 0; j < num_threads; j++)
        {
            const unsigned int min_r = j * block_size;
            const unsigned int max_r = std::min((j + 1) * block_size, h);
            boost::asio::post(pool,
                              std::bind(segment_block, 0, w, min_r, max_r,
                                        color_similarities, features, bg_colors,
                                        colored_frame, std::ref(gray_frame),
                                        w));
        }

        // Wait for all threads to finish
        pool.join();

        // Segment the current frame based on the color and texture similarities
        // with the background frame
        segmentation_helper::segment(*color_similarities, bg_features,
                                     *features, w, result, colored_bg_frame,
                                     colored_frame, learning_rate, weights);

        // Log the duration of the segmentation
        if (verbose)
        {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(end
                                                                      - start)
                    .count();

            BOOST_LOG_TRIVIAL(info) << "Frame " << i << "/" << size
                                    << " segmented in " << duration << "ms";
        }

        // Free the memory
        delete color_similarities;
        delete features;
    }

    /**
     * Extract the color and texture features from the background frame
     * @param w The width of the frame
     * @param h The height of the frame
     * @param colored_bg_frame The background frame in color
     * @param bg_colors The background color components
     * @param bg_features The texture features of the background frame
     */
    void extract_frame(const unsigned int w, const unsigned int h,
                       cv::Mat* colored_bg_frame,
                       color_helper::color_vectors* bg_colors,
                       texture_helper::feature_vector* bg_features)
    {
        // Convert the background frame to grayscale
        cv::Mat gray_bg_frame;
        cv::cvtColor(*colored_bg_frame, gray_bg_frame, cv::COLOR_BGR2GRAY);

        // Calculate the center value for LBP
        uint8_t center = gray_bg_frame.at<uint8_t>(0, 0);

        // Process the first row
        unsigned int c = 0;
        for (; c < w; c++)
        {
            // Calculate LBP for the pixel in the row
            (*bg_features)[c] =
                texture_helper::calculateLBP(gray_bg_frame, c, 0);

            // Update the color components from the background frame for the
            // given
            color_helper::convert(*colored_bg_frame, c, 0, bg_colors->at(0)[c],
                                  bg_colors->at(1)[c]);
        }

        // Process the remaining rows
        unsigned int r = 1;
        for (; r < h; r++)
        {
            uint8_t previous_row_pixel = (*bg_features)[(r - 1) * w];

            // Calculate LBP for the first pixel in the row
            (*bg_features)[r * w] =
                texture_helper::calculateLBP(gray_bg_frame, 0, r);

            // Update the color components from the background frame for the
            // given
            color_helper::convert(*colored_bg_frame, 0, r,
                                  bg_colors->at(0)[r * w],
                                  bg_colors->at(1)[r * w]);

            // Process the remaining pixels in the row
            c = 1;
            for (; c < w; c++)
            {
                uint8_t current_pixel =
                    texture_helper::calculateLBP(gray_bg_frame, c, r);

                // Shift the previous row's pixel value to the left by 1 bit
                previous_row_pixel <<= 1;

                // Update the previous row's pixel value with the current pixel
                // value
                previous_row_pixel |= (current_pixel < center);

                // Set the LBP value for the current pixel in the features array
                (*bg_features)[r * w + c] = previous_row_pixel;

                // Update the color components from the background frame for the
                // given
                color_helper::convert(*colored_bg_frame, c, r,
                                      bg_colors->at(0)[r * w + c],
                                      bg_colors->at(1)[r * w + c]);
            }
        }

        // Release the gray background frame
        gray_bg_frame.release();
    }

} // namespace segmentation_helper