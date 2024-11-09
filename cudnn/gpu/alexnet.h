#ifndef ALEXNET_H
#define ALEXNET_H

#include <cudnn.h>

class AlexNet {
public:
    AlexNet(cudnnHandle_t& handle, int batch_size);
    ~AlexNet();
    
    // Forward declaration of network structure
    void createNetwork();

    void forward(float *inp, float *out);
    
private:
    cudnnHandle_t& cudnn;
    int batch_size;
    
    // Layer descriptors
    cudnnTensorDescriptor_t input_descriptor;
    cudnnFilterDescriptor_t conv1_filter_descriptor;
    cudnnConvolutionDescriptor_t conv1_descriptor;
    // ... other layer descriptors will follow
    
    // Weights and biases
    float *conv1_weights, *conv1_bias;
    // ... other weights and biases will follow
    
    void createConv1();
    void createPool1();
    // ... other layer creation methods will follow
};

#endif // ALEXNET_H 