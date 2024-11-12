#include "alexnet.h"
#include <cstdlib>
#include "utils.h"

Network::Network(cudnnHandle_t& handle, int batch_size, int num_classes_) 
    : cudnn(handle), batch_size_(batch_size), num_classes_(num_classes_) {
    cublasCreate(&cublas);
}

void Network::forward(float *inp, float *out) {
    float* current_input = inp;
    
    for (size_t i = 0; i < layers.size(); i++) {
        // TODO: This is a hack to get the output of the last layer. should be layers.size() - 1
        // 
        float* current_output = (i == layers.size() - 1) ? out : layer_outputs[i];
        layers[i]->forward(current_input, current_output);
        current_input = current_output;
    }
}

float* Network::createDummyGradient(float* output) {
    // Create a gradient of ones, similar to PyTorch's ones_like
    size_t output_dim = batch_size_ * num_classes_;
    float* gradient;
    cudaMallocManaged(&gradient, output_dim * sizeof(float));
    
    for (size_t i = 0; i < output_dim; i++) {
        gradient[i] = 1.0f;
    }
    return gradient;
}

void Network::backwardInput(float* inp_grad, float* out_grad) {
    float* current_output_grad = out_grad;
    
    // Backward pass through layers in reverse order
    for (int i = layers.size() - 1; i >= 0; i--) {
        // Determine where to store the computed gradient
        float* current_input_grad;
        if (i == 0) {
            current_input_grad = inp_grad;
        } else {
            current_input_grad = gradient_outputs[i-1];  // Store in gradient_outputs instead of layer_outputs
        }

        // Compute gradients
        if (auto* conv_layer = dynamic_cast<ConvolutionLayer*>(layers[i])) {
            conv_layer->backwardInput(current_input_grad, current_output_grad);
        } else if (auto* fc_layer = dynamic_cast<FCLayer*>(layers[i])) {
            fc_layer->backwardInput(current_input_grad, current_output_grad);
        }

        // Update for next iteration
        current_output_grad = current_input_grad;
    }
}

void Network::backwardParams(float* inp, float* out_grad) {
    float* current_input = inp;
    float* current_output_grad = out_grad;
    
    // Compute parameter gradients for each layer
    for (int i = layers.size() - 1; i >= 0; i--) {
        if (auto* conv_layer = dynamic_cast<ConvolutionLayer*>(layers[i])) {
            // Handle convolutional layer
            conv_layer->backwardParams(current_input, current_output_grad);
        } else if (auto* fc_layer = dynamic_cast<FCLayer*>(layers[i])) {
            // Handle fully connected layer
            fc_layer->backwardParams(current_input, current_output_grad);
        }
        
        if (i > 0) {
            current_input = layer_outputs[i-1];
            current_output_grad = layer_outputs[i-1];
        }
    }
}

void Network::updateWeights(float learning_rate) {
    float lr = -learning_rate;  // Negative because cublasSaxpy does addition

    for (int i = layers.size() - 1; i >= 0; i--) {
        ConvolutionLayer* conv = static_cast<ConvolutionLayer*>(layers[i]);
        // Update conv1 weights
        cublasSaxpy(cublas,
                    conv->getWeightSize(),
                    &lr,
                    conv->getWeightGradients(), 1,
                    conv->getWeights(), 1);

        checkWeightChanges("Conv1", conv->getWeights(), conv->getWeightSize());
    }
}

void Network::zeroGradients() {
    for (Layer* layer : layers) {
        layer->zeroGradients();
    }
}

Network::~Network() {
    // Destroy cublas handle
    cublasDestroy(cublas);
    
    // Free layer objects
    for (Layer* layer : layers) {
        delete layer;
    }
    
    // Free forward pass outputs
    for (float* output : layer_outputs) {
        if (output != nullptr) {
            cudaFree(output);
        }
    }

    // Free gradient outputs
    for (float* gradient : gradient_outputs) {
        if (gradient != nullptr) {
            cudaFree(gradient);
        }
    }
}

void Network::addConvLayer(int width, int height, int in_channels, int out_channels, 
                          int kernel_size, int stride, int padding) {
    // Create and add the convolutional layer
    layers.push_back(new ConvolutionLayer(cudnn, batch_size_, 
                                        in_channels, out_channels, 
                                        kernel_size, stride, padding));
    
    // Allocate memory for layer outputs and gradients
    size_t output_size = batch_size_ * out_channels * width * height * sizeof(float);  // Adjust dimensions based on input/stride/padding
    
    // Forward pass output
    float* layer_output;
    cudaMalloc(&layer_output, output_size);
    layer_outputs.push_back(layer_output);

    // Backward pass gradient
    float* gradient_output;
    cudaMalloc(&gradient_output, output_size);
    gradient_outputs.push_back(gradient_output);
}

void Network::addFCLayer(int in_features, int out_features) {
    // Create and add the fully connected layer
    layers.push_back(new FCLayer(cudnn, batch_size_, in_features, out_features));
    
    // Allocate memory for layer outputs and gradients
    size_t output_size = batch_size_ * out_features * sizeof(float);
    
    // Forward pass output
    float* layer_output;
    cudaMalloc(&layer_output, output_size);
    layer_outputs.push_back(layer_output);

    // Backward pass gradient
    float* gradient_output;
    cudaMalloc(&gradient_output, output_size);
    gradient_outputs.push_back(gradient_output);
}