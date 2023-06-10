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

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int batchSize = 10;
    std::vector<cv::Mat> frames(batchSize);

    while (true) {
        size_t i = 0;
        for (; i < batchSize && cap.read(frame); ++i) {
            frames[i] = frame.clone();
        }
        if (i == 0) 
            break;  

        frames[0] = frame.clone();
        for (int i = 1; i < batchSize && cap.read(frame); ++i) {
            frames[i] = frame.clone();
        }

        int frameCount = 0;
        auto start = std::chrono::high_resolution_clock::now();

        auto blockSize = dim3(128, 128);
        auto gridSize = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        for (size_t i = 0; i < frames.size(); ++i) {

            cudaMemcpyAsync(d_image2, frames[i].ptr<uchar3>(), width * height * sizeof(uchar3), cudaMemcpyHostToDevice, stream);

            fuzzy_integral << <gridSize, blockSize, 0, stream >> > (d_image1, d_image2, d_lbpBackground, d_result, width, height);

            cv::Mat processed_frame(height, width, CV_32F);
            cudaMemcpyAsync(processed_frame.ptr<float>(), d_result, width * height * sizeof(float), cudaMemcpyDeviceToHost, stream);

            cudaStreamSynchronize(stream);

            std::swap(d_image1, d_image2);

            cv::Mat output_frame;
            processed_frame.convertTo(output_frame, CV_8UC1, 255.0);
            writer.write(output_frame);

            frameCount++;
            if (frameCount % 100 == 0) { 
                auto now = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start);
                std::cout << "Frames per second: " << static_cast<double>(frameCount) / duration.count() << std::endl;
            }
        }
    }

    cudaFree(d_image1);
    cudaFree(d_image2);
    cudaFree(d_lbpBackground);
    cudaFree(d_result);
    cudaStreamDestroy(stream);

    cap.release();
    writer.release();
}

