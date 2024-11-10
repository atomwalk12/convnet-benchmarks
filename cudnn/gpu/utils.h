#ifndef CUDNN_UTILS_H
#define CUDNN_UTILS_H

#include <cudnn.h>

// Debug function for tensor descriptors
void debugDescriptor(const char* name, cudnnTensorDescriptor_t desc);

// Debug function for filter descriptors
void debugFilterDescriptor(cudnnFilterDescriptor_t desc);

#endif // CUDNN_UTILS_H 