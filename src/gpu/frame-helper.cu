#include "frame-helper.cuh"
#include "segmentation-helper.cuh"
#include <chrono>

void process_frames(const std::string& input_path, const std::string& output_path) {
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Unable to open the file" << std::endl;
        exit(1);
    }

    cv::Mat frame;
    cap.read(frame);
    cv::Size frameSize = frame.size();

    int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
    double fps = 30.0;
    cv::VideoWriter writer(output_path, fourcc, fps, frameSize, true);

    if (!writer.isOpened()) {
        std::cerr << "Unable to open the output file for writing" << std::endl;
        exit(1);
    }

    int width = frame.cols;
    int height = frame.rows;
    uchar3* d_image1;
    uchar3* d_image2;
    uint8_t* d_lbpBackground;
    float* d_result;
    cudaMalloc(&d_image1, width * height * sizeof(uchar3));
    cudaMalloc(&d_image2, width * height * sizeof(uchar3));
    cudaMalloc(&d_lbpBackground, width * height * sizeof(uint8_t));
    cudaMalloc(&d_result, width * height * sizeof(float));
    cudaMemcpy(d_image1, frame.ptr<uchar3>(), width * height * sizeof(uchar3), cudaMemcpyHostToDevice);

    uint8_t* h_lbpBackground = new uint8_t[width * height];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_lbpBackground[y * width + x] = calculateLBP(frame.ptr<uchar3>(), x, y, width, height);
        }
    }

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaMemcpyAsync(d_lbpBackground, h_lbpBackground, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice, stream1);

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    int frameCount = 0;
    auto start = std::chrono::high_resolution_clock::now();
    bool isFrameRead = true;

    do {
        frameCount++;
        if (frameCount % 100 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start);
            std::cout << "Frames per second: " << static_cast<double>(frameCount) / duration.count() << std::endl;
        }

        cudaMemcpyAsync(d_image2, frame.ptr<uchar3>(), width * height * sizeof(uchar3), cudaMemcpyHostToDevice, stream1);
        fuzzy_integral<<<gridSize, blockSize, 0, stream1>>>(d_image1, d_image2, d_lbpBackground, d_result, width, height);

        if (isFrameRead) {
            isFrameRead = cap.read(frame);
            cudaMemcpyAsync(d_image1, d_image2, width * height * sizeof(uchar3), cudaMemcpyDeviceToDevice, stream2);
        }

        cv::Mat processed_frame(height, width, CV_32F);
        cudaMemcpyAsync(processed_frame.ptr<float>(), d_result, width * height * sizeof(float), cudaMemcpyDeviceToHost, stream1);
        cudaStreamSynchronize(stream1);

        cv::Mat output_frame;
        processed_frame.convertTo(output_frame, CV_8UC1, 255.0);
        writer.write(output_frame);
    } while (isFrameRead);

    delete[] h_lbpBackground;
    cudaFree(d_image1);
    cudaFree(d_image2);
    cudaFree(d_lbpBackground);
    cudaFree(d_result);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cap.release();
    writer.release();
}
