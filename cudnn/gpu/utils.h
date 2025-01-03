#ifndef UTILS_H
#define UTILS_H

#include <cudnn.h>

// Debug configuration
extern bool ENABLE_DEBUG_OUTPUT;  // Declaration

// Existing function declarations
void debugDescriptor(const char* name, cudnnTensorDescriptor_t desc);
void debugFilterDescriptor(cudnnFilterDescriptor_t desc);
void debugTensorValues(const char* label, float* device_ptr, int count);
void checkWeightChanges(const char* label, float* device_weights, int size);
void debugMatrixLayout(const char* label, float* device_ptr, 
                      int rows, int cols, int max_rows = 3, int max_cols = 5);

#endif // UTILS_H 