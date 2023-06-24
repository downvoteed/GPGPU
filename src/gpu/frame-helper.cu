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

    // resolution
    std::cout << "Resolution: " << frameSize.width << "x" << frameSize.height << std::endl;
    // fps
	std::cout << "FPS: " << cap.get(cv::CAP_PROP_FPS) << std::endl;
	// number of frames
	std::cout << "Number of frames: " << cap.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;

    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    double fps = cap.get(cv::CAP_PROP_FPS);
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

    // Calculate LBP of the first frame and copy it to the GPU
    uint8_t* h_lbpBackground = new uint8_t[width * height];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_lbpBackground[y * width + x] = calculateLBP(frame.ptr<uchar3>(), x, y, width, height);
        }
    }

    cudaMemcpyAsync(d_lbpBackground, h_lbpBackground, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    bool isFrameRead = true;

    // Launch kernel asynchronously with two streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

	auto tStart = std::chrono::steady_clock::now();
	auto tNow = tStart;
    auto toriginal = tStart;
	int frameCount = 0;


	do {
		++frameCount;
		// Time since start of current second
		auto timeFromStartMs = std::chrono::duration_cast<std::chrono::milliseconds>(tNow - tStart);
		if (timeFromStartMs.count() >= 1000)
        { 
			std::cout << "Frames per second: " << frameCount << std::endl;
			frameCount = 0; 
			tStart = tNow; 
		}

        cudaMemcpyAsync(d_image2, frame.ptr<uchar3>(), width * height * sizeof(uchar3), cudaMemcpyHostToDevice, stream1);
        fuzzy_integral<<<gridSize, blockSize, 0, stream1>>>(d_image1, d_image2, d_lbpBackground, d_result, width, height);

        if (isFrameRead) {
            isFrameRead = cap.read(frame);
            cudaMemcpyAsync(d_image1, d_image2, width * height * sizeof(uchar3), cudaMemcpyDeviceToDevice, stream2);
        }

        cv::Mat processed_frame(height, width, CV_32F);
        cudaMemcpy2DAsync(processed_frame.ptr<float>(), width * sizeof(float), d_result, width * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost, stream1);
        cudaStreamSynchronize(stream1);

        processed_frame.convertTo(processed_frame, CV_8UC1, 255.0);
        cv::cvtColor(processed_frame, processed_frame, cv::COLOR_GRAY2BGR);
        writer.write(processed_frame);
		tNow = std::chrono::steady_clock::now();

    } while (isFrameRead);

    // compute average fps
    tNow = std::chrono::steady_clock::now();
    std::cout << "Average FPS: " << cap.get(cv::CAP_PROP_FRAME_COUNT) / std::chrono::duration_cast<std::chrono::milliseconds>(tNow - toriginal).count() * 1000 << std::endl;

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
