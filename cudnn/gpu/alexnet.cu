#include "alexnet.h"
#include <cstdlib>

AlexNet::AlexNet(cudnnHandle_t& handle, int batch_size, int num_classes) 
    : cudnn(handle), batch_size(batch_size), output_size(num_classes) {
    createNetwork();
}

void AlexNet::createNetwork() {
    // First convolution layer: 96 kernels of 11x11, stride 4
    layers.push_back(new ConvolutionLayer(cudnn, batch_size, 3, 96, 11, 4, 0));
    
    // Allocate memory for intermediate outputs
    // We'll need to calculate the output size for each layer
    // This is a placeholder - actual sizes need to be computed
    layer_outputs.push_back(nullptr);  // Will be set in forward pass
}

void AlexNet::forward(float *inp, float *out) {
    float* current_input = inp;
    
    for (size_t i = 0; i < layers.size(); i++) {
        float* current_output = (i == layers.size() - 1) ? out : layer_outputs[i];
        layers[i]->forward(current_input, current_output);
        current_input = current_output;
    }
}

float* AlexNet::createDummyGradient(float* output) {
    // Create a gradient of ones, similar to PyTorch's ones_like
    size_t output_dim = batch_size * output_size; // hardcoded for now, should match output size
    float* gradient;
    cudaMallocManaged(&gradient, output_dim * sizeof(float));
    
    for (size_t i = 0; i < output_dim; i++) {
        gradient[i] = 1.0f;
    }
    return gradient;
}

void AlexNet::backwardInput(float* inp_grad, float* out_grad) {
    float* current_output_grad = out_grad;
    
    // Backward pass through layers in reverse order
    for (int i = layers.size() - 1; i >= 0; i--) {
        ConvolutionLayer* conv_layer = static_cast<ConvolutionLayer*>(layers[i]);
        float* current_input_grad = (i == 0) ? inp_grad : layer_outputs[i-1];
        conv_layer->backwardInput(current_input_grad, current_output_grad);
        current_output_grad = current_input_grad;
    }
}

void AlexNet::backwardParams(float* inp, float* out_grad) {
    float* current_input = inp;
    float* current_output_grad = out_grad;
    
    // Compute parameter gradients for each layer
    for (int i = layers.size() - 1; i >= 0; i--) {
        ConvolutionLayer* conv_layer = static_cast<ConvolutionLayer*>(layers[i]);
        conv_layer->backwardParams(current_input, current_output_grad);
        
        // Update input and gradient for next layer
        if (i > 0) {
            current_input = layer_outputs[i-1];
            current_output_grad = layer_outputs[i-1]; // This will be overwritten by backwardInput
        }
    }
}

AlexNet::~AlexNet() {
    // Delete all layers
    for (Layer* layer : layers) {
        delete layer;
    }
    
    // Free intermediate outputs
    for (float* output : layer_outputs) {
        if (output != nullptr) {
            cudaFree(output);
        }
    }
}