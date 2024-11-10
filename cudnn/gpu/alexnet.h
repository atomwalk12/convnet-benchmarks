#ifndef ALEXNET_H
#define ALEXNET_H

#include <cudnn.h>
#include <vector>
#include "layer.h"
#include "conv_layer.h"

class AlexNet {
public:
    AlexNet(cudnnHandle_t& handle, int batch_size, int output_size);
    ~AlexNet();
    
    void createNetwork();
    void forward(float *inp, float *out);
    float* createDummyGradient(float* output);
    void backwardInput(float *inp_grad, float *out_grad);
    void backwardParams(float *inp, float *out_grad);
    
private:
    cudnnHandle_t& cudnn;
    int batch_size;
    
    // Vector to store all layers
    std::vector<Layer*> layers;
    
    // Intermediate outputs between layers
    std::vector<float*> layer_outputs;

    int output_size;
};

#endif // ALEXNET_H 