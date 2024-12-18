#include "relu.h"
#include <cstdio>


__global__
void relu_forward_gpu(float *inp, float *out, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    if (ind < sz_out){
        out[ind] = fmaxf(0, inp[ind]);
    }
}


__global__
void relu_backward_gpu(float* gradient_out, float* gradient_in, float* forward_input, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    if (ind < sz_out){
        gradient_out[ind] = (forward_input[ind] > 0) ? gradient_in[ind] : 0;
    }
}


ReLU_GPU::ReLU_GPU(int _sz_out){
    sz_out = _sz_out;

    n_blocks = (sz_out + block_size - 1) / block_size;
}


void ReLU_GPU::forward(float* _inp, float* _out){
    forward_input = _inp;
    relu_forward_gpu<<<n_blocks, block_size>>>(_inp, _out, sz_out);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in ReLU forward: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}


void ReLU_GPU::backward(float* gradient_out, float* gradient_in) {
    relu_backward_gpu<<<n_blocks, block_size>>>(gradient_out, gradient_in, forward_input, sz_out);
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in ReLU backward: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}
