#include "loss.h"
#include <stdio.h>

// CUDA kernels
__global__ void mse_forward_kernel(float* prediction, float* target, float* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = prediction[idx] - target[idx];
        loss[idx] = diff * diff;
    }
}

__global__ void mse_backward_kernel(float* prediction, float* target, float* gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // d(MSE)/dx = 2(x - y)/n
        gradient[idx] = 2.0f * (prediction[idx] - target[idx]) / size;
    }
}


float MSELoss::compute(float* prediction, float* target, int size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    // Allocate temporary buffer dynamically
    float* d_buffer;
    cudaMallocManaged(&d_buffer, size * sizeof(float));
    
    // Compute squared differences
    mse_forward_kernel<<<num_blocks, block_size>>>(
        prediction, target, d_buffer, size
    );
    
    // Get result
    float total_loss = 0.0f;
    cudaMemcpy(&total_loss, d_buffer, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_buffer);
    
    return total_loss / size;
}

void MSELoss::backward(float* prediction, float* target, float* gradient, int size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    mse_backward_kernel<<<num_blocks, block_size>>>(
        prediction, target, gradient, size
    );
} 