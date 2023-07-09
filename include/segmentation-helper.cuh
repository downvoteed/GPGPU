#include <stdint.h>

__device__ __host__ uint8_t calculateLBP(const uchar3* image, int idx, int idy,
                                         int width, int height);

__device__ float compare(uint8_t lbp1, uint8_t lbp2);

__global__ void fuzzy_integral(uchar3* image1, uchar3* image2,
                               uint8_t* lbpBackground, float* result, int width,
                               int height);

__global__ void calculate_lbp_kernel(uchar3* image, float* result, int width,
                                     int height);
