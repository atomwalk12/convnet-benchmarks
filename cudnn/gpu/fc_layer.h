#ifndef FC_LAYER_H
#define FC_LAYER_H

#include <cudnn.h>
#include <cublas_v2.h>
#include "layer.h"

class FCLayer : public Layer {
public:
    FCLayer(cudnnHandle_t& cudnn_handle, 
            int batch_size,
            int input_features,
            int output_features);
    ~FCLayer();

    void createDescriptors() override;
    void forward(float* input, float* output) override;
    void backwardInput(float* input_gradient, float* output_gradient) override;
    void backwardParams(float* input, float* output_gradient) override;
    void destroyDescriptors() override;
    void zeroGradients() override;

    float* getWeights() override { return weights; }
    float* getWeightGradients() override { return weight_gradients; }
    int getWeightSize() override { 
        return input_features * output_features; 
    }

private:
    int batch_size;
    int input_features;
    int output_features;
    
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    
    float *weights;
    float *weight_gradients;
    
    cublasHandle_t cublas_handle;
};

#endif // FC_LAYER_H 