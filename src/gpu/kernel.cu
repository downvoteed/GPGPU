#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
#include <stdio.h>

__global__ void helloCUDA(float f)
{
	printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}

__device__ __host__ uint8_t calculateLBP(uchar3* image, int idx, int idy, int width, int height) {
    float center = 0.2989f * image[idy * width + idx].x + 0.5870f * image[idy * width + idx].y + 0.1140f * image[idy * width + idx].z;

    uint8_t lbp = 0;

    int directions[8][2] = { {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} };

    for (int i = 0; i < 8; i++) {
        int x = min(max(idx + directions[i][0], 0), width - 1);
        int y = min(max(idy + directions[i][1], 0), height - 1);

        float value = 0.2989f * image[y * width + x].x + 0.5870f * image[y * width + x].y + 0.1140f * image[y * width + x].z;

        lbp = (lbp << 1) | (value < center);
    }

    return lbp;
}

__device__ float compare(uint8_t lbp1, uint8_t lbp2) {
    uint8_t vector = ~(lbp1 ^ lbp2);
    return __popc(vector) / 8.0f;
}

__global__ void fuzzy_integral_and_LBP(uchar3* image1, uchar3* image2, uint8_t* lbpBackground, float* result, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx >= width || idy >= height) return;

    uchar3 pixel1 = image1[idy * width + idx];
    uchar3 pixel2 = image2[idy * width + idx];

    float r_ratio, g_ratio;
    if (pixel1.x > pixel2.x) {
        r_ratio = (float)pixel2.x / (float)pixel1.x;
    } else {
        r_ratio = (float)pixel1.x / (float)pixel2.x;
    }

    if (pixel1.y > pixel2.y) {
        g_ratio = (float)pixel2.y / (float)pixel1.y;
    } else {
        g_ratio = (float)pixel1.y / (float)pixel2.y;
    }

    // print pixels values
     //printf("%d %d %d %d %d %d\n", pixel1.x, pixel1.y, pixel1.z, pixel2.x, pixel2.y, pixel2.z);

    uint8_t lbp1 = lbpBackground[idy * width + idx];
    uint8_t lbp2 = calculateLBP(image2, idx, idy, width, height);

    float lbp_similarity = compare(lbp1, lbp2);

    float coefficients[] = { r_ratio, g_ratio, lbp_similarity };
    float temp;

    for(int i=0; i<2; ++i){
        for(int j=0; j<2-i; ++j){
            if(coefficients[j] > coefficients[j+1]){
                temp = coefficients[j];
                coefficients[j] = coefficients[j+1];
                coefficients[j+1] = temp;
            }
        }
    }

    float final_result = coefficients[0] * 0.1f + coefficients[1] * 0.3f + coefficients[2] * 0.6f;
   // printf("%f %f %f %f\n", coefficients[0], coefficients[1], coefficients[2], final_result);

    result[idy * width + idx] = final_result > 0.67f ? 0 : 255;
}


int main(int argc, char** argv)
{
    std::string input_path = "C:\\Users\\mouis\\EPITA\\GPGPU\\gpgpu\\dataset\\video.avi";
    std::string output_path = "C:\\Users\\mouis\\EPITA\\GPGPU\\gpgpu\\dataset\\outputGPU.mp4";

    std::vector<cv::Mat> frames;
    cv::VideoCapture cap(input_path);

    if (!cap.isOpened())
    {
        std::cerr << "Unable to open the file" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::Size frameSize;
	while (cap.read(frame))
	{
		frameSize = frame.size();
		frames.push_back(frame.clone());
	}


    int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
    double fps = 30.0;
    cv::VideoWriter writer(output_path, fourcc, fps, frameSize, true);

    if (!writer.isOpened())
    {
        std::cerr << "Unable to open the output file for writing" << std::endl;
        return -1;
    }

    int width = frames[0].cols;
    int height = frames[0].rows;
    uchar3* d_image1;
    uchar3* d_image2;
    float* d_result;
    cudaMalloc(&d_image1, width * height * sizeof(uchar3));
    cudaMalloc(&d_image2, width * height * sizeof(uchar3));
    cudaMalloc(&d_result, width * height * sizeof(float));

    // Calculate and copy LBP of the first frame to the GPU
    cv::Mat firstFrame = frames[0];
    uchar3* h_image1 = firstFrame.ptr<uchar3>();
    uint8_t* h_lbpBackground = new uint8_t[width * height];
    for(int y=0; y<height; ++y){
        for(int x=0; x<width; ++x){
            h_lbpBackground[y * width + x] = calculateLBP(h_image1, x, y, width, height);
        }
    }
    uint8_t* d_lbpBackground;
    cudaMalloc(&d_lbpBackground, width * height * sizeof(uint8_t));
    cudaMemcpy(d_lbpBackground, h_lbpBackground, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

    for (size_t i = 1; i < frames.size(); i++)
    {
        cudaMemcpy(d_image1, firstFrame.ptr<uchar3>(), width * height * sizeof(uchar3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_image2, frames[i].ptr<uchar3>(), width * height * sizeof(uchar3), cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        fuzzy_integral_and_LBP<<<gridSize, blockSize>>>(d_image1, d_image2, d_lbpBackground, d_result, width, height);

        cv::Mat processed_frame(height, width, CV_32F);
        cudaMemcpy(processed_frame.ptr<float>(), d_result, width * height * sizeof(float), cudaMemcpyDeviceToHost);

        processed_frame.convertTo(processed_frame, CV_8UC1, 255.0);
        cv::cvtColor(processed_frame, processed_frame, cv::COLOR_GRAY2BGR);
        writer.write(processed_frame);
    }

    cudaFree(d_image1);
    cudaFree(d_image2);
    cudaFree(d_result);
    cudaFree(d_lbpBackground);

    cap.release();
    writer.release();

    return 0;
}
