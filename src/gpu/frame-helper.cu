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
    cv::Size frameSize;
    cap.read(frame);
    frameSize = frame.size();

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
    cudaMallocManaged(&d_image1, width * height * sizeof(uchar3)); 
    cudaMallocManaged(&d_image2, width * height * sizeof(uchar3));
	float* d_result;
	cudaMallocManaged(&d_result, width * height * sizeof(float));

    // background
    uchar3* h_image1 = frame.ptr<uchar3>();
    uint8_t* h_lbpBackground = new uint8_t[width * height];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_lbpBackground[y * width + x] = calculateLBP(h_image1, x, y, width, height);
        }
    }
    uint8_t* d_lbpBackground;
    cudaMallocManaged(&d_lbpBackground, width * height * sizeof(uint8_t));
    cudaMemcpy(d_lbpBackground, h_lbpBackground, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaMemcpyAsync(d_image1, frame.ptr<uchar3>(), width * height * sizeof(uchar3), cudaMemcpyHostToDevice, stream1);

    while (cap.read(frame))
    {
        cudaMemcpyAsync(d_image2, frame.ptr<uchar3>(), width * height * sizeof(uchar3), cudaMemcpyHostToDevice, stream1);

        dim3 blockSize(32, 32);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        fuzzy_integral<<<gridSize, blockSize, 0, stream1>>>(d_image1, d_image2, d_lbpBackground, d_result, width, height);

        cv::Mat frame_output(height, width, CV_8UC1); 
        cudaMemcpyAsync(frame_output.ptr<uint8_t>(), d_result, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream2); // changed from float to uint8_t

        std::swap(d_image1, d_image2);  

        cudaStreamSynchronize(stream2); 

        cv::cvtColor(frame_output, frame_output, cv::COLOR_GRAY2BGR);
        writer.write(frame_output);
    }

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaFree(d_image1);
    cudaFree(d_image2);
    cudaFree(d_result);
    cudaFree(d_lbpBackground);

    cap.release();
    writer.release();
}

