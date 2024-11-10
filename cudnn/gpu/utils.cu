#include "utils.h"
#include <stdio.h>

// Debug configuration
bool ENABLE_DEBUG_OUTPUT = false;  // Definition with default value

void debugDescriptor(const char* name, cudnnTensorDescriptor_t desc) {
    if (!ENABLE_DEBUG_OUTPUT) return;
    cudnnDataType_t dtype;
    int n, c, h, w;
    int stride_n, stride_c, stride_h, stride_w;
    
    cudnnGetTensor4dDescriptor(desc,
        &dtype,
        &n, &c, &h, &w,
        &stride_n, &stride_c, &stride_h, &stride_w);
        
    printf("%s descriptor:\n", name);
    printf("  Dimensions: %dx%dx%dx%d\n", n, c, h, w);
    printf("  Strides: %d,%d,%d,%d\n", stride_n, stride_c, stride_h, stride_w);
    fflush(stdout);
}

void debugFilterDescriptor(cudnnFilterDescriptor_t desc) {
    if (!ENABLE_DEBUG_OUTPUT) return;
    cudnnDataType_t dtype;
    cudnnTensorFormat_t format;
    int k, c, h, w;
    
    cudnnGetFilter4dDescriptor(desc,
        &dtype,
        &format,
        &k, &c, &h, &w);
        
    printf("Filter descriptor:\n");
    printf("  Dimensions: %dx%dx%dx%d\n", k, c, h, w);
    fflush(stdout);
}

void debugTensorValues(const char* label, float* device_ptr, int count) {
    if (!ENABLE_DEBUG_OUTPUT) return;
    
    float debug_values[10];  // Static size for simplicity
    int print_count = std::min(count, 10);  // Print at most 10 values
    
    cudaError_t err = cudaMemcpy(debug_values, device_ptr, 
                                print_count * sizeof(float), 
                                cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess) {
        printf("Error copying %s values: %s\n", 
               label, cudaGetErrorString(err));
        return;
    }

    printf("First %d %s values: ", print_count, label);
    for(int i = 0; i < print_count; i++) {
        printf("%.4f ", debug_values[i]);
    }
    printf("\n");
    fflush(stdout);
}

void checkWeightChanges(const char* label, float* device_weights, int size) {

    static float prev_sum = 0.0f;  // Keep track of previous sum
    
    // Copy weights to host
    float* host_weights = new float[size];
    cudaMemcpy(host_weights, device_weights, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute simple statistics
    float sum = 0.0f;
    for(int i = 0; i < size; i++) {
        sum += host_weights[i];
    }
    
    printf("%s - Weight sum: %.4f (change: %.4f)\n", 
           label, sum, sum - prev_sum);
    fflush(stdout);
    prev_sum = sum;
    
    delete[] host_weights;
} 