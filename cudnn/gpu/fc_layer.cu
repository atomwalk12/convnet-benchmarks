#include "fc_layer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cudnn.h>
#include <random>


FCLayer::FCLayer(cudnnHandle_t& cudnn_handle,
                 int batch_size,
                 int input_features,
                 int output_features)
    : Layer(cudnn_handle),
      batch_size(batch_size),
      input_features(input_features),
      output_features(output_features) {
    
    cublasCreate(&cublas_handle);
    createDescriptors();
    
    // Allocate and initialize weights
    size_t weight_size = input_features * output_features * sizeof(float);
    cudaMallocManaged(&weights, weight_size);
    cudaMallocManaged(&weight_gradients, weight_size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // He initialization - using input_features as fan_in
    float std = sqrt(2.0f / input_features);
    std::normal_distribution<float> distribution(0.0f, std);

    for (int i = 0; i < input_features * output_features; i++) {
        weights[i] = distribution(gen);
    }
    
    zeroGradients();
}

void FCLayer::createDescriptors() {
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnCreateTensorDescriptor(&output_descriptor);
    
    cudnnSetTensor4dDescriptor(
        input_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, input_features, 1, 1
    );
    
    cudnnSetTensor4dDescriptor(
        output_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, output_features, 1, 1
    );
}

void FCLayer::forward(float* input, float* output) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Perform matrix multiplication: output = weights * input
    // Goal: output = weights × input
    cublasStatus_t status = cublasSgemm(cublas_handle,
                CUBLAS_OP_N,        // weights as-is: [output_features × input_features]
                CUBLAS_OP_N,        // input as-is: [input_features × batch_size]
                output_features,    // rows of A
                batch_size,         // cols of A
                input_features,     // cols of A and rows of B
                &alpha,
                weights,            // [output_features × input_features]
                output_features,    
                input,              // [input_features × batch_size]
                input_features,
                &beta,
                output,             // [output_features × batch_size]
                output_features);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS error: %s\n", cublasGetStatusString(status));
        exit(1);
    }

    // Debug output values
    float* host_output = new float[output_features * batch_size];
    cudaMemcpy(host_output, output, output_features * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
}

void FCLayer::backwardInput(float* input_gradient, float* output_gradient) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Perform matrix multiplication: input_gradient = weights^T × output_gradient
    cublasStatus_t status = cublasSgemm(cublas_handle,
                CUBLAS_OP_T,        // Need weights transposed
                CUBLAS_OP_N,        // output_gradient as-is
                input_features,     // rows of result
                batch_size,         // cols of result
                output_features,    // inner dimension
                &alpha,
                weights,            // [output_features × input_features]
                output_features,
                output_gradient,    // [output_features × batch_size]
                output_features,
                &beta,
                input_gradient,     // [input_features × batch_size]
                input_features);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS error: %s\n", cublasGetStatusString(status));
        exit(1);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

void FCLayer::backwardParams(float* input, float* output_gradient) {
    const float alpha = 1.0f;
    const float beta = 1.0f;  // Accumulate gradients


    // Perform matrix multiplication: weight_gradients = output_gradient × input^T
    // See https://docs.nvidia.com/cuda/cublas/ -> cublasSgemm
    cublasStatus_t status = cublasSgemm(cublas_handle,
                CUBLAS_OP_N,        // output_gradient as-is
                CUBLAS_OP_T,        // Need input transposed
                output_features,    // rows of result
                input_features,     // cols of result
                batch_size,         // inner dimension
                &alpha,
                output_gradient,    // [output_features × batch_size]
                output_features,
                input,              // [input_features × batch_size]
                input_features,
                &beta,
                weight_gradients,   // [output_features × input_features]
                output_features);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS error: %s\n", cublasGetStatusString(status));
        exit(1);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

void FCLayer::zeroGradients() {
    cudaMemset(weight_gradients, 0, getWeightSize() * sizeof(float));
}

void FCLayer::destroyDescriptors() {
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudaFree(weights);
    cudaFree(weight_gradients);
}

FCLayer::~FCLayer() {
    destroyDescriptors();
    cublasDestroy(cublas_handle);
}