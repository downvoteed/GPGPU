#include "frame-helper.cuh"
#include "segmentation-helper.cuh"


void process_frames(const std::string& input_path, const std::string& output_path) {
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened())
    {
        std::cerr << "Unable to open the file" << std::endl;
        exit(1);
    }

    cv::Mat frame;
    cap.read(frame);
    cv::Size frameSize = frame.size();

    int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
    double fps = 30.0;
    cv::VideoWriter writer(output_path, fourcc, fps, frameSize, true);

    if (!writer.isOpened())
    {
        std::cerr << "Unable to open the output file for writing" << std::endl;
        exit(1);
    }

    int width = frame.cols;
    int height = frame.rows;
    uchar3* d_image1;
    uchar3* d_image2;
    float* d_result;
    cudaMalloc(&d_image1, width * height * sizeof(uchar3));
    cudaMalloc(&d_image2, width * height * sizeof(uchar3));
    cudaMalloc(&d_result, width * height * sizeof(float));

    // Calculate LBP of the first frame and copy it to the GPU
    uint8_t* h_lbpBackground = new uint8_t[width * height];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_lbpBackground[y * width + x] = calculateLBP(frame.ptr<uchar3>(), x, y, width, height);
        }
    }

    uint8_t* d_lbpBackground;
    cudaMalloc(&d_lbpBackground, width * height * sizeof(uint8_t));
    cudaMemcpy(d_lbpBackground, h_lbpBackground, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    do {
        cudaMemcpy(d_image2, frame.ptr<uchar3>(), width * height * sizeof(uchar3), cudaMemcpyHostToDevice);

        fuzzy_integral<<<gridSize, blockSize>>>(d_image1, d_image2, d_lbpBackground, d_result, width, height);

        cv::Mat processed_frame(height, width, CV_32F);
        cudaMemcpy(processed_frame.ptr<float>(), d_result, width * height * sizeof(float), cudaMemcpyDeviceToHost);

        std::swap(d_image1, d_image2);

        processed_frame.convertTo(processed_frame, CV_8UC1, 255.0);
        cv::cvtColor(processed_frame, processed_frame, cv::COLOR_GRAY2BGR);
        writer.write(processed_frame);

    } while (cap.read(frame));

    cudaFree(d_image1);
    cudaFree(d_image2);
    cudaFree(d_result);
    cudaFree(d_lbpBackground);

    cap.release();
    writer.release();
}

