#include "utils.h"
#include "conv_layer.h"
#include <stdio.h>
#include <cublas_v2.h>
#include <random>
#include <cmath>

ConvolutionLayer::ConvolutionLayer(cudnnHandle_t& cudnn_handle,
                                 int input_width,
                                 int input_height,
                                 int batch_size,
                                 int in_channels,
                                 int out_channels,
                                 int kernel_size,
                                 int stride,
                                 int padding)
    : Layer(cudnn_handle),
      batch_size(batch_size),
      in_channels(in_channels),
      out_channels(out_channels),
      kernel_size(kernel_size),
      stride(stride),
      padding(padding),
      input_height(input_height),
      input_width(input_width) {

    createDescriptors();
    calculateOutputDimensions();

    // Initialize cublas
    cublasCreate(&cublas_handle);
}

void ConvolutionLayer::createDescriptors() {   
    // Input descriptor
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(
        input_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, in_channels, input_height, input_width
    );

    // Filter descriptor
    cudnnCreateFilterDescriptor(&filter_descriptor);
    cudnnSetFilter4dDescriptor(
        filter_descriptor,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        out_channels, in_channels, kernel_size, kernel_size
    );

    // Convolution descriptor
    cudnnCreateConvolutionDescriptor(&conv_descriptor);
    cudnnSetConvolution2dDescriptor(
        conv_descriptor,
        padding,
        padding,
        stride,
        stride,
        1,  // dilation_h
        1,  // dilation_w
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    );

    // Calculate output dimensions
    int out_n, out_c, out_h, out_w;
    cudnnGetConvolution2dForwardOutputDim(
        conv_descriptor,
        input_descriptor,
        filter_descriptor,
        &out_n,
        &out_c,
        &out_h,
        &out_w
    );

    // Store output dimensions as class members
    output_height = out_h;  // Add to header
    output_width = out_w;   // Add to header

    // Output descriptor
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(
        output_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, out_channels, output_height, output_width
    );

    // Allocate and initialize weights and biases
    size_t weight_size = getWeightSize();
    cudaMallocManaged(&weights, weight_size * sizeof(float));
    cudaMallocManaged(&weight_gradients, weight_size * sizeof(float));

    // Initialize weights using Gaussian distribution (AlexNet paper specification)
    std::random_device rd;
    std::mt19937 gen(rd());
    float std = sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    float std_alexnet = 0.01f;
    std::normal_distribution<float> distribution(0.0f, std);

    for (size_t i = 0; i < weight_size; i++) {
        weights[i] = distribution(gen);
    }

    debugDescriptor("Input", input_descriptor);
    debugDescriptor("Output", output_descriptor);
    debugFilterDescriptor(filter_descriptor);

    // Debug print first few weights
    debugTensorValues("weights", weights, 10);

    // Get workspace size needed for the selected algorithm
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        algo,
        &workspace_size
    );

    // Allocate workspace memory
    workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }
}

void ConvolutionLayer::forward(float* input, float* output) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform convolution using the pre-allocated workspace
    cudnnStatus_t status = cudnnConvolutionForward(
        cudnn,
        &alpha,
        input_descriptor,
        input,
        filter_descriptor,
        weights,
        conv_descriptor,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        workspace,
        workspace_size,
        &beta,
        output_descriptor,
        output
    );
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("CUDNN forward failed: %s\n", cudnnGetErrorString(status));
        exit(1);
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Debug output values
    debugTensorValues("output", output, 10);
}

void ConvolutionLayer::destroyDescriptors() {
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(conv_descriptor);
    cudaFree(weights);
    cudaFree(bias);
    cudaFree(weight_gradients);
}

void ConvolutionLayer::backwardInput(float* input_gradient, float* output_gradient) {
    // Debug output gradient values
    debugTensorValues("output gradient", output_gradient, 10);
    
    // Verify pointers
    if (input_gradient == nullptr || output_gradient == nullptr || weights == nullptr) {
        printf("Error: Null pointer in backwardInput\n");
        exit(1);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Get workspace size needed
    size_t workspace_size = 0;
    cudnnStatus_t status = cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn,
        filter_descriptor,
        output_descriptor,
        conv_descriptor,
        input_descriptor,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        &workspace_size
    );

    if (status != CUDNN_STATUS_SUCCESS) {
        printf("Error getting workspace size: %s\n", cudnnGetErrorString(status));
        exit(1);
    }

    // Allocate workspace
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t err = cudaMalloc(&workspace, workspace_size);
        if (err != cudaSuccess) {
            printf("Workspace allocation failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }
    }

    // Backward pass with different algorithm
    // Implements: dL/dx = (dL/dy) @ w
    status = cudnnConvolutionBackwardData(
        cudnn,
        &alpha,
        filter_descriptor,
        weights,
        output_descriptor,
        output_gradient,
        conv_descriptor,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        workspace,        // Added workspace
        workspace_size,   // Added workspace size
        &beta,
        input_descriptor,
        input_gradient
    );

    if (status != CUDNN_STATUS_SUCCESS) {
        printf("Backward data failed: %s\n", cudnnGetErrorString(status));
        exit(1);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Free workspace
    if (workspace) {
        cudaFree(workspace);
    }

    // Debug input gradient values
    debugTensorValues("input gradient", input_gradient, 10);
}

void ConvolutionLayer::backwardParams(float* input, float* output_gradient) {
    // Debug input and gradient values
    debugTensorValues("input", input, 10);
    debugTensorValues("output gradient", output_gradient, 10);
    
    // First verify pointers
    if (input == nullptr || output_gradient == nullptr || weight_gradients == nullptr) {
        printf("Error: Null pointer passed to backwardParams\n");
        return;
    }
    // Zero out gradients before computing new ones
    // TODO to delete
    // zeroGradients();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Implements: dL/dW = x ⊗ (dL/dy)
    cudnnStatus_t status = cudnnConvolutionBackwardFilter(
        cudnn,
        &alpha,
        input_descriptor,
        input,
        output_descriptor,
        output_gradient,
        conv_descriptor,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        nullptr,
        0,
        &beta,
        filter_descriptor,
        weight_gradients
    );

    if (status != CUDNN_STATUS_SUCCESS) {
        printf("Backward filter failed: %s\n", cudnnGetErrorString(status));
        exit(1);
    }

    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after backward filter: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Debug weight gradients
    debugTensorValues("weight gradients", weight_gradients, 10);
}

void ConvolutionLayer::zeroGradients() {
    cudaMemset(weight_gradients, 0, getWeightSize() * sizeof(float));
}

ConvolutionLayer::~ConvolutionLayer() {
    destroyDescriptors();
    // Destroy cublas handle
    cublasDestroy(cublas_handle);
    if (workspace) {
        cudaFree(workspace);
    }
}

