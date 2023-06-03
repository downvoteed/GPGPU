#include "cuda_runtime.h"

#include <stdio.h>

// main hello world


__global__ void helloCUDA(float f)
{
	printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}

int main(int argc, char** argv)
{
	printf("Hello CUDA\n");
	helloCUDA<<<1, 5>>>(1.2345f);
	cudaDeviceReset();
	return 0;
}
