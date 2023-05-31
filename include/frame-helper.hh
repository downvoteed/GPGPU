#pragma once

#include <opencv2/opencv.hpp>

namespace frame_helper
{
    using frames = std::vector<cv::Mat>;

    /**
     * Read a frame from a file
     * @param path The path to the file
     * @param gray Whether to read the frame in grayscale
     * @return The OpenCV frame
     */
    cv::Mat readFrame(const std::string &path, bool gray = false)
    {
        int flag = cv::IMREAD_COLOR;
        if (gray)
        {
            flag = cv::IMREAD_GRAYSCALE;
        }

        cv::Mat frame = cv::imread(path, flag);
        if (frame.empty())
        {
            std::cout << "Could not read the image: " << path << std::endl;
            exit(1);
        }

        return frame;
    }

    /**
     * Show a frame in a window
     * @param name The name of the window
     * @param frame The OpenCV frame
     */
    void showFrame(const std::string &name, const cv::Mat &frame)
    {
        cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
        cv::imshow(name, frame);
        cv::waitKey(0);
        cv::destroyWindow(name);
    }

    /**
     * Build a segmented frame from a vector of boolean values (foreground/background)
     * @param segments The vector of boolean values (1 or 0)
     * @param width The width of the frame
     * @param height The height of the frame
     * @return The segmented frame
     */
    cv::Mat buildSegmentedFrame(const std::vector<uint8_t> &segments, int width, int height)
    {
        cv::Mat frame = cv::Mat::zeros(height, width, CV_8UC1);
        for (int i = 0; i < segments.size(); i++)
        {
            int x = i % width;
            int y = i / width;
            frame.at<uint8_t>(y, x) = segments[i] * 255;
        }

        return frame;
    }

    /**
     * Read a video file and return its frames
     * @param path The path to the video file
     * @param gray Whether to read the frames in grayscale
     * @return The frames
     */
    frames readFrames(const std::string &path, bool gray = false)
    {
        cv::VideoCapture video(path);
        if (!video.isOpened())
        {
            std::cout << "Error opening video stream or file" << std::endl;
            exit(1);
        }

        frames frames;
        while (true)
        {
            cv::Mat frame;
            video >> frame;

            if (frame.empty())
            {
                break;
            }

            if (gray)
            {
                cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
            }

            cv::resize(frame, frame, cv::Size(1500, 1500));
            frames.push_back(frame);
        }

        video.release();
        return frames;
    }

} // namespace frame_helper
