#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "layer.h"
#include <cublas_v2.h>

class ConvolutionLayer : public Layer {
public:
    ConvolutionLayer(cudnnHandle_t& cudnn_handle, 
                     int input_width,
                     int input_height,
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
    void destroyDescriptors() override;
    void zeroGradients() override;

    float* getWeights() { return weights; }
    float* getWeightGradients() { return weight_gradients; }
    int getWeightSize() { 
        return out_channels * in_channels * kernel_size * kernel_size; 
    }
    int getOutputHeight() { return output_height; }
    int getOutputWidth() { return output_width; }
    int getOutChannels() { return out_channels; }

    void calculateOutputDimensions() {
        output_height = ((input_height + 2*padding - kernel_size) / stride) + 1;
        output_width = ((input_width + 2*padding - kernel_size) / stride) + 1;
    }

    size_t getInputSize() const {
        return batch_size * in_channels * input_height * input_width;
    }


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
    
    cublasHandle_t cublas_handle;
    void* workspace;
    size_t workspace_size;
};

#endif // CONV_LAYER_H 