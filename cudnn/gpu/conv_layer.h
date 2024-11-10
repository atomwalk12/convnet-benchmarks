#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "layer.h"

class ConvolutionLayer : public Layer {
public:
    ConvolutionLayer(cudnnHandle_t& cudnn_handle, 
                     int batch_size,
                     int in_channels,
                     int out_channels,
                     int kernel_size,
                     int stride,
                     int padding);
    ~ConvolutionLayer();

    void createDescriptors() override;
    void forward(float* input, float* output) override;
    void backwardInput(float* input_gradient, float* output_gradient);
    void backwardParams(float* input, float* output_gradient);
    void updateWeights(float learning_rate) override;
    void destroyDescriptors() override;

private:
    int batch_size, in_channels, out_channels;
    int kernel_size, stride, padding;
    int input_height, input_width;
    int output_height, output_width;
    
    cudnnTensorDescriptor_t input_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnConvolutionDescriptor_t conv_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    
    float *weights, *bias;
    float *weight_gradients, *bias_gradients;
};

#endif // CONV_LAYER_H 