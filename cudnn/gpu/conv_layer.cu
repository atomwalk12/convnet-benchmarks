#include "utils.h"
#include "conv_layer.h"
#include <stdio.h>

ConvolutionLayer::ConvolutionLayer(cudnnHandle_t& cudnn_handle,
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
      padding(padding) {
    createDescriptors();
}

void ConvolutionLayer::createDescriptors() {
    // Add these as class members to track dimensions
    input_height = 224;  // Add to header
    input_width = 224;   // Add to header
    
    // Input descriptor
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(
        input_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        in_channels,
        input_height,
        input_width
    );

    // Filter descriptor
    cudnnCreateFilterDescriptor(&filter_descriptor);
    cudnnSetFilter4dDescriptor(
        filter_descriptor,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        out_channels,
        in_channels,
        kernel_size,
        kernel_size
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
        out_n,
        out_c,
        out_h,
        out_w
    );

    // Allocate and initialize weights and biases
    size_t weight_size = out_channels * in_channels * kernel_size * kernel_size * sizeof(float);
    cudaMallocManaged(&weights, weight_size);
    cudaMallocManaged(&bias, out_channels * sizeof(float));
    cudaMallocManaged(&weight_gradients, weight_size);

    // Initialize weights with random values
    for (size_t i = 0; i < weight_size/sizeof(float); i++) {
        weights[i] = (float)rand() / RAND_MAX;
    }

    // Initialize bias to zero
    for (int i = 0; i < out_channels; i++) {
        bias[i] = 0.0f;
    }

    // Debug print
    printf("Input dimensions: %dx%dx%dx%d\n", batch_size, in_channels, input_height, input_width);
    printf("Output dimensions: %dx%dx%dx%d\n", out_n, out_c, output_height, output_width);
    

    debugDescriptor("Input", input_descriptor);
    debugDescriptor("Output", output_descriptor);
    debugFilterDescriptor(filter_descriptor);
    fflush(stdout);
}

void ConvolutionLayer::forward(float* input, float* output) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Implements: y = w ⊗ x
    // To simplify things, biases are ignored
    cudnnConvolutionForward(
        cudnn,
        &alpha,
        input_descriptor,
        input,
        filter_descriptor,
        weights,
        conv_descriptor,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        nullptr,  // workspace (we should add this as a class member)
        0,        // workspace size
        &beta,
        input_descriptor,
        output
    );
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
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Implements: dL/dx = w ⊗ (dL/dy)
    cudnnConvolutionBackwardData(
        cudnn,
        &alpha,
        filter_descriptor,  // w descriptor
        weights,            // w
        input_descriptor,   // dy descriptor
        output_gradient,    // dy
        conv_descriptor,    // convolution descriptor
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        nullptr,           // workspace
        0,                 // workspace size
        &beta,
        input_descriptor,  // dx descriptor
        input_gradient     // dx
    );
}

void ConvolutionLayer::backwardParams(float* input, float* output_gradient) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Implements: dL/dW = x ⊗ (dL/dy)
    cudnnConvolutionBackwardFilter(
        cudnn,
        &alpha,
        input_descriptor,    // x descriptor
        input,               // x
        input_descriptor,    // dy descriptor
        output_gradient,     // dy
        conv_descriptor,     // convolution descriptor
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        nullptr,            // workspace
        0,                  // workspace size
        &beta,
        filter_descriptor,  // dw descriptor
        weight_gradients    // dw
    );
}

void ConvolutionLayer::updateWeights(float learning_rate) {
    // Simple SGD update
    int weight_size = out_channels * in_channels * kernel_size * kernel_size;
    for(int i = 0; i < weight_size; i++) {
        weights[i] -= learning_rate * weight_gradients[i];
    }
}

ConvolutionLayer::~ConvolutionLayer() {
    destroyDescriptors();
}

