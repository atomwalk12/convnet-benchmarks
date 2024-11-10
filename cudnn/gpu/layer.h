#ifndef LAYER_H
#define LAYER_H

#include <cudnn.h>

class Layer {
public:
    Layer(cudnnHandle_t& cudnn_handle) : cudnn(cudnn_handle) {}
    virtual ~Layer() = default;
    
    virtual void createDescriptors() = 0;
    virtual void forward(float* input, float* output) = 0;
    virtual void backwardInput(float* input_gradient, float* output_gradient) = 0;
    virtual void backwardParams(float* input, float* output_gradient) = 0;
    virtual void updateWeights(float learning_rate) = 0;
    virtual void destroyDescriptors() = 0;

protected:
    cudnnHandle_t& cudnn;
};

#endif // LAYER_H 