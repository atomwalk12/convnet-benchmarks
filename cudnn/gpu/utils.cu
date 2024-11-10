#include "utils.h"
#include <stdio.h>

void debugDescriptor(const char* name, cudnnTensorDescriptor_t desc) {
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