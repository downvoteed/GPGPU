#include "webcam-processor.hh"

/**
 * Process the webcam stream
 * @param verbose Whether to display the log messages
 * @param num_threads The number of threads to use
 * @param learning_rate The learning_rate value for the background optimizer
 */
void process_webcam(const bool verbose, const std::optional<unsigned int> width,
                    const std::optional<unsigned int> height,
                    const bool flipped, const unsigned int num_threads,
                    const double learning_rate)
{
    const double lr = learning_rate == 0 ? 0.1 : learning_rate;

    // Open the webcam flipped
    cv::VideoCapture webcam(0);

    // Set the webcam properties
    if (width.has_value())
    {
        webcam.set(cv::CAP_PROP_FRAME_WIDTH, width.value());
    }
    if (height.has_value())
    {
        webcam.set(cv::CAP_PROP_FRAME_HEIGHT, height.value());
    }

    // Check if the webcam was opened successfully
    if (!webcam.isOpened())
    {
        std::cerr << "Failed to open the webcam!" << std::endl;
        exit(1);
    }

    // Get the width and height of the webcam stream
    const unsigned int w = webcam.get(cv::CAP_PROP_FRAME_WIDTH);
    const unsigned int h = webcam.get(cv::CAP_PROP_FRAME_HEIGHT);

    if (verbose)
    {
        BOOST_LOG_TRIVIAL(info) << "Webcam stream opened";
        BOOST_LOG_TRIVIAL(info) << "- Frame size: " << w << "x" << h;
    }

    // Create the window to display the webcam stream
    cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);

    // Store the background frames and features
    cv::Mat* colored_bg_frame = nullptr;
    color_helper::color_vectors* bg_colors = new color_helper::color_vectors(
        { color_helper::color_vector(w * h, 0),
          color_helper::color_vector(w * h, 0) });
    texture_helper::feature_vector* bg_features =
        new texture_helper::feature_vector(w * h, 0);

    // Create a matrix to store the weights of the background features
    cv::Mat weights(w, h, CV_32FC1, cv::Scalar(0.5));

    // Keep track of the average execution time
    auto total_duration = std::chrono::high_resolution_clock::duration::zero();
    unsigned int total_frames = 0;

    const bool should_extract_bg = true;

    // Main loop
    while (true)
    {
        // Read a frame from the webcam
        cv::Mat frame;
        webcam >> frame;

        // Flip the frame if needed
        if (flipped)
        {
            cv::flip(frame, frame, 1);
        }

        // Convert the frame to grayscale
        cv::Mat gray_frame;
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

        // If ESC or Q is pressed, exit the loop
        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q')
        {
            // Release the frame
            frame.release();
            gray_frame.release();

            break;
        }

        // If no background features are available, extract them from the frame
        if (should_extract_bg)
        {
            if (verbose)
            {
                BOOST_LOG_TRIVIAL(info) << "Extracting the background frame";
            }

            // Set the background frames
            if (colored_bg_frame == nullptr)
            {
                colored_bg_frame = new cv::Mat(frame.clone());
            }

            // Extract the background features
            segmentation_helper::extract_frame(w, h, colored_bg_frame,
                                               bg_colors, bg_features);

            if (verbose)
            {
                BOOST_LOG_TRIVIAL(info) << "Background features extracted";
            }
        }

        // Start a timer to measure the execution time
        auto start = std::chrono::high_resolution_clock::now();

        // Segment the frame
        cv::Mat* result = new cv::Mat(h, w, CV_8UC1);
        segmentation_helper::segment_frame(
            0, 0, colored_bg_frame, *bg_features, *bg_colors, frame, gray_frame,
            w, h, false, std::ref(*result), num_threads, lr, weights);

        // Display the frame
        cv::imshow("Webcam", *result);

        if (verbose)
        {
            // Stop the timer
            auto stop = std::chrono::high_resolution_clock::now();

            // Calculate the elapsed time
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(stop
                                                                      - start);

            // Update the average execution time
            total_duration += duration;
            total_frames++;

            BOOST_LOG_TRIVIAL(info) << "Frame segmented in " << duration.count()
                                    << "ms ("
                                    << total_frames
                    / std::chrono::duration_cast<std::chrono::seconds>(
                          total_duration)
                          .count() << "fps)";
        }

        // Free the matrix
        frame.release();
        gray_frame.release();

        result->release();
        delete result;
    }

    // Release the webcam
    webcam.release();

    // Release the window
    cv::destroyWindow("Webcam");

    // Release the colored background frame
    colored_bg_frame->release();

    // Free the memory
    delete bg_colors;
    delete bg_features;
    delete colored_bg_frame;
}
