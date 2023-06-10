#include "segmentation-helper.cuh"

__global__ void calculate_lbp_kernel(uchar3* image, float* result, int width, int height) {
	__shared__ int directions[8][2];  // declare as shared memory

	if (threadIdx.x == 0 && threadIdx.y == 0) {  // let the first thread initialize the shared data
		directions[0][0] = -1; directions[0][1] = -1;
		directions[1][0] = 0;  directions[1][1] = -1;
		directions[2][0] = 1;  directions[2][1] = -1;
		directions[3][0] = 1;  directions[3][1] = 0;
		directions[4][0] = 1;  directions[4][1] = 1;
		directions[5][0] = 0;  directions[5][1] = 1;
		directions[6][0] = -1; directions[6][1] = 1;
		directions[7][0] = -1; directions[7][1] = 0;
	}

	__syncthreads();  // make sure all threads in a block have the shared data before computation

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx < width && idy < height) {
		float center = 0.2989f * image[idy * width + idx].x + 0.5870f * image[idy * width + idx].y + 0.1140f * image[idy * width + idx].z;

		uint8_t lbp = 0;

		for (int i = 0; i < 8; i++) {
			int x = min(max(idx + directions[i][0], 0), width - 1);
			int y = min(max(idy + directions[i][1], 0), height - 1);

			float value = 0.2989f * image[y * width + x].x + 0.5870f * image[y * width + x].y + 0.1140f * image[y * width + x].z;

			lbp = (lbp << 1) | (value < center);
		}

		result[idy * width + idx] = lbp;
	}
}

__device__ __host__ uint8_t calculateLBP(const uchar3* image, int idx, int idy, int width, int height) {
    float center = 0.2989f * image[idy * width + idx].x + 0.5870f * image[idy * width + idx].y + 0.1140f * image[idy * width + idx].z;

    uint8_t lbp = 0;

    // Unroll loop for 8 directions
    int x, y;
    float value;

    // direction: {-1,-1}
    x = min(max(idx - 1, 0), width - 1);
    y = min(max(idy - 1, 0), height - 1);
    value = 0.2989f * image[y * width + x].x + 0.5870f * image[y * width + x].y + 0.1140f * image[y * width + x].z;
    lbp |= static_cast<uint8_t>(value < center) << 0;

    // direction: {0,-1}
    x = min(max(idx, 0), width - 1);
    y = min(max(idy - 1, 0), height - 1);
    value = 0.2989f * image[y * width + x].x + 0.5870f * image[y * width + x].y + 0.1140f * image[y * width + x].z;
    lbp |= static_cast<uint8_t>(value < center) << 1;

    // direction: {1,-1}
    x = min(max(idx + 1, 0), width - 1);
    y = min(max(idy - 1, 0), height - 1);
    value = 0.2989f * image[y * width + x].x + 0.5870f * image[y * width + x].y + 0.1140f * image[y * width + x].z;
    lbp |= static_cast<uint8_t>(value < center) << 2;

    // direction: {1,0}
    x = min(max(idx + 1, 0), width - 1);
    y = min(max(idy, 0), height - 1);
    value = 0.2989f * image[y * width + x].x + 0.5870f * image[y * width + x].y + 0.1140f * image[y * width + x].z;
    lbp |= static_cast<uint8_t>(value < center) << 3;

    // direction: {1,1}
    x = min(max(idx + 1, 0), width - 1);
    y = min(max(idy + 1, 0), height - 1);
    value = 0.2989f * image[y * width + x].x + 0.5870f * image[y * width + x].y + 0.1140f * image[y * width + x].z;
    lbp |= static_cast<uint8_t>(value < center) << 4;

    // direction: {0,1}
    x = min(max(idx, 0), width - 1);
    y = min(max(idy + 1, 0), height - 1);
    value = 0.2989f * image[y * width + x].x + 0.5870f * image[y * width + x].y + 0.1140f * image[y * width + x].z;
    lbp |= static_cast<uint8_t>(value < center) << 5;

    // direction: {-1,1}
    x = min(max(idx - 1, 0), width - 1);
    y = min(max(idy + 1, 0), height - 1);
    value = 0.2989f * image[y * width + x].x + 0.5870f * image[y * width + x].y + 0.1140f * image[y * width + x].z;
    lbp |= static_cast<uint8_t>(value < center) << 6;

    // direction: {-1,0}
    x = min(max(idx - 1, 0), width - 1);
    y = min(max(idy, 0), height - 1);
    value = 0.2989f * image[y * width + x].x + 0.5870f * image[y * width + x].y + 0.1140f * image[y * width + x].z;
    lbp |= static_cast<uint8_t>(value < center) << 7;

    return lbp;
}


__device__ float compare(uint8_t lbp1, uint8_t lbp2) {
    uint8_t vector = ~(lbp1 ^ lbp2);
    return __popc(vector) / 8.0f;
}

__global__ void fuzzy_integral(uchar3* image1, uchar3* image2, uint8_t* lbpBackground, float* result, int width, int height) {
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
    result[idy * width + idx] = final_result > 0.67f ? 0.0f : 255.0f;
}

