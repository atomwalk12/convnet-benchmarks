#include "fc_layer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cudnn.h>


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
    
    // Xavier initialization
    float scale = sqrt(2.0f / (input_features + output_features));
    for (int i = 0; i < input_features * output_features; i++) {
        weights[i] = scale * ((float)rand() / RAND_MAX * 2 - 1);
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
    cublasStatus_t status = cublasSgemm(cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                output_features,  // m: rows of output
                batch_size,      // n: cols of output
                input_features,  // k: cols of weights
                &alpha,
                weights,
                output_features,
                input,
                input_features,
                &beta,
                output,
                output_features);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS error: %s\n", cublasGetStatusString(status));
        exit(1);
    }
}

void FCLayer::backwardInput(float* input_gradient, float* output_gradient) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Perform matrix multiplication: input_gradient = weights^T * output_gradient
    cublasStatus_t status = cublasSgemm(cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                input_features,  // m: rows of input gradient
                batch_size,     // n: cols of input gradient
                output_features, // k: rows of weights
                &alpha,
                weights,
                input_features, // lda: leading dimension of weights (was output_features)
                output_gradient,
                output_features, // ldb: leading dimension of output_gradient
                &beta,
                input_gradient,
                input_features); // ldc: leading dimension of input_gradient

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS error: %s\n", cublasGetStatusString(status));
        exit(1);
    }
}

void FCLayer::backwardParams(float* input, float* output_gradient) {
    const float alpha = 1.0f;
    const float beta = 1.0f;  // Accumulate gradients

    printf("Feature-related variables:\n");
    printf("output_features: %d\n", output_features);
    printf("input_features: %d\n", input_features);

    printf("\nGradient and data pointers:\n");
    printf("output_gradient pointer: %p\n", output_gradient);
    printf("input pointer: %p\n", input);
    printf("weight_gradients pointer: %p\n", weight_gradients);
    fflush(stdout);
    // Perform matrix multiplication: weight_gradients = output_gradient * input^T
    // https://www-inf.telecom-sudparis.eu/COURS/IA307/IA307-course4-IntroductionToCuBLAS.pdf
    cublasStatus_t status = cublasSgemm(cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                output_features,    // m: rows of weight gradients
                input_features,     // n: cols of weight gradients
                batch_size,         // k: inner dimension for multiplication
                &alpha,
                output_gradient,    // [output_features × batch_size]
                output_features,    // lda: leading dimension of output_gradient
                input,             // [batch_size × input_features]
                batch_size,        // ldb: leading dimension of input (was input_features)
                &beta,
                weight_gradients,   // [output_features × input_features]
                output_features);   // ldc: leading dimension of weight_gradients

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS error: %s\n", cublasGetStatusString(status));
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