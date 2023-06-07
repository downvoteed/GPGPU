#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"

#include <stdio.h>

__global__ void helloCUDA(float f)
{
	printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}

__global__ void fuzzy_integral(float* image, float* result, int width, int height) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	float sum = 0;
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			int x = min(max(idx + i, 0), width - 1);
			int y = min(max(idy + j, 0), height - 1);
			sum += image[y * width + x];
		}
	}
	result[idy * width + idx] = sum / 8.0f;
}


int main(int argc, char** argv)
{
	printf("Hello CUDA\n");
	helloCUDA << <1, 5 >> > (1.2345f);

	std::string input_path = "C:\\Users\\mouis\\EPITA\\GPGPU\\gpgpu\\dataset\\video.avi";
	std::string output_path = "C:\\Users\\mouis\\EPITA\\GPGPU\\gpgpu\\dataset\\output.mp4";

	std::vector<cv::Mat> frames;

	cv::VideoCapture cap(input_path);

	if (!cap.isOpened())
	{
		std::cerr << "Unable to open the file" << std::endl;
		return -1;
	}

	// first frame
	cv::Mat first_frame;
	cap >> first_frame;

	while (true)
	{
		cv::Mat frame;
		cap >> frame;
		if (frame.empty())
			break;

		// Convert the frame to grayscale and float32
		cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
		frame.convertTo(frame, CV_32F);

		frames.push_back(frame);
	}


	int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
	double fps = 30.0;
	cv::VideoWriter writer(output_path, fourcc, fps, first_frame.size());

	if (!writer.isOpened())
	{
		std::cerr << "Unable to open the output file for writing" << std::endl;
		return -1;
	}

	int width = frames[0].cols;
	int height = frames[0].rows;
	float* d_image;
	float* d_result;
	cudaMalloc(&d_image, width * height * sizeof(float));
	cudaMalloc(&d_result, width * height * sizeof(float));

	for (cv::Mat& frame : frames)
	{
		cudaMemcpy(d_image, frame.ptr<float>(), width * height * sizeof(float), cudaMemcpyHostToDevice);

		dim3 blockSize(16, 16);
		dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
		fuzzy_integral << <gridSize, blockSize >> > (d_image, d_result, width, height);

		cv::Mat processed_frame(height, width, CV_32F);
		cudaMemcpy(processed_frame.ptr<float>(), d_result, width * height * sizeof(float), cudaMemcpyDeviceToHost);

		cv::threshold(processed_frame, processed_frame, 128, 255, cv::THRESH_BINARY);
		cv::Mat bgr_processed_frame;
		cv::cvtColor(processed_frame, bgr_processed_frame, cv::COLOR_GRAY2BGR);
		bgr_processed_frame.convertTo(bgr_processed_frame, CV_8UC3);

		writer << bgr_processed_frame;
	}

	cudaFree(d_image);
	cudaFree(d_result);
	cudaDeviceReset();


}
