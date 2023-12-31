#include "video-processor.hh"

/**
 * Process a video file
 * @param verbose Whether to display the log messages
 * @param video_path The path to the video file
 * @param width The width of the frames
 * @param height The height of the frames
 * @param output_path The path to the output video file
 * @param num_threads The number of threads to use
 * @param display Whether to display the segmented frames
 * @param fps The FPS of the output video file
 * @param learning_rate The learning_rate value for the background optimizer
 */
void process_video(const bool verbose, const std::string& video_path,
                   const std::optional<unsigned int> width,
                   const std::optional<unsigned int> height,
                   const std::optional<std::string> output_path,
                   const unsigned int num_threads, const bool display,
                   const unsigned int fps, const double learning_rate)
{
    // Start a timer to measure the execution time
    const auto start = std::chrono::high_resolution_clock::now();

    if (verbose)
    {
        BOOST_LOG_TRIVIAL(info) << "Loading dataset";
    }

    const frame_helper::frames_vector* frames_vector =
        frame_helper::readFrames(video_path, width, height);
    const frame_helper::frames& colored_frames = frames_vector->at(0);
    const frame_helper::frames& gray_frames = frames_vector->at(1);

    // log RAM usage and display it
    if (verbose)
    {
        // logger.begin_rss_loging();
    }

    if (colored_frames.size() == 0)
    {
        std::cerr << "No colored frames found!" << std::endl;
        exit(1);
    }

    if (gray_frames.size() == 0)
    {
        std::cerr << "No grayscale frames found!" << std::endl;
        exit(1);
    }

    if (verbose)
    {
        BOOST_LOG_TRIVIAL(info) << "Dataset loaded";
        BOOST_LOG_TRIVIAL(info) << "- " << colored_frames.size() << " frames";
    }

    // Extract the first frame from the dataset as the background
    cv::Mat* colored_bg_frame = new cv::Mat(colored_frames[0]);
    const unsigned int w = colored_bg_frame->cols;
    const unsigned int h = colored_bg_frame->rows;

    if (verbose)
    {
        BOOST_LOG_TRIVIAL(info) << "- Frame size: " << w << "x" << h;
        BOOST_LOG_TRIVIAL(info) << "Extracting the background frame";
    }

    // Extract the color components and texture features from the background
    // frame
    color_helper::color_vectors* bg_colors =
        new color_helper::color_vectors{ color_helper::color_vector(w * h, 0),
                                         color_helper::color_vector(w * h, 0) };
    texture_helper::feature_vector* bg_features =
        new texture_helper::feature_vector(w * h, 0);

    // Extract the background features
    segmentation_helper::extract_frame(w, h, colored_bg_frame, bg_colors,
                                       bg_features);

    // Create a matrix to store the weights of the background features
    cv::Mat weights(w, h, CV_32FC1, cv::Scalar(0.5));

    auto end = std::chrono::high_resolution_clock::now();

    if (verbose)
    {
        BOOST_LOG_TRIVIAL(info)
            << "Background frame extracted in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end
                                                                     - start)
                   .count()
            << "ms";
    }

    boost::asio::thread_pool pool(num_threads);

    // Process the frames
    frame_helper::frames_ref segmented_frames = {};
    for (unsigned long i = 1; i < colored_frames.size(); i++)
    {
        // Create a new frame to store the result
        cv::Mat* result = new cv::Mat(h, w, CV_8UC1);

        // Update the background features
        if (learning_rate > 0)
        {
            // Segment the frame
            segmentation_helper::segment_frame(
                i, colored_frames.size(), colored_bg_frame, *bg_features,
                *bg_colors, colored_frames[i], gray_frames[i], w, h, verbose,
                std::ref(*result), num_threads, learning_rate,
                std::ref(weights));

            // Extract the background features
            segmentation_helper::extract_frame(w, h, colored_bg_frame,
                                               bg_colors, bg_features);

            // Free the memory by dropping the const
            const_cast<cv::Mat&>(colored_frames[i]).release();
            const_cast<cv::Mat&>(gray_frames[i]).release();
        }
        else
        {
            boost::asio::post(
                pool,
                [i, colored_bg_frame, bg_features, bg_colors, colored_frames,
                 gray_frames, w, h, verbose, result, num_threads, learning_rate,
                 &weights]() {
                    // Segment the frame
                    segmentation_helper::segment_frame(
                        i, colored_frames.size(), colored_bg_frame,
                        *bg_features, *bg_colors, colored_frames[i],
                        gray_frames[i], w, h, verbose, std::ref(*result),
                        num_threads, learning_rate, std::ref(weights));

                    // Release the frames
                    const_cast<cv::Mat&>(colored_frames[i]).release();
                    const_cast<cv::Mat&>(gray_frames[i]).release();
                });
        }

        segmented_frames.push_back(result);
    }

    // Wait for all the threads to finish
    pool.join();

    if (verbose)
    {
        end = std::chrono::high_resolution_clock::now();
        const auto total_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();

        BOOST_LOG_TRIVIAL(info)
            << "Processing completed in " << total_duration << "ms";
    }

    // Save the segmented frames to a video file
    if (output_path.has_value())
    {
        if (verbose)
        {
            BOOST_LOG_TRIVIAL(info)
                << "Saving the segmented frames to " << output_path.value();
        }

        frame_helper::saveFrames(output_path.value(), segmented_frames, fps);

        if (verbose)
        {
            BOOST_LOG_TRIVIAL(info) << "Segmented frames saved";
        }
    }

    // Display the segmented frames one by one and wait for a key press to
    // display the next one
    if (display)
    {
        for (unsigned long i = 0; i < segmented_frames.size(); i++)
        {
            frame_helper::showFrame(
                "Segmented Frame " + std::to_string(i) + "/"
                    + std::to_string(segmented_frames.size()),
                *(segmented_frames[i]));
        }
    }

    // Release the frames
    colored_bg_frame->release();

    // Free the memory
    delete frames_vector;
    delete bg_colors;
    delete bg_features;
    delete colored_bg_frame;

    for (unsigned long i = 0; i < segmented_frames.size(); i++)
    {
        segmented_frames[i]->release();
        delete segmented_frames[i];
    }
}