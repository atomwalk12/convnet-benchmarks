#ifndef LAYER_H
#define LAYER_H

class Layer {
public:
    Layer(cudnnHandle_t& cudnn_handle) : cudnn(cudnn_handle) {}
    virtual ~Layer() = default;
    
    virtual void createDescriptors() = 0;
    virtual void forward(float* input, float* output) = 0;
    virtual void backwardInput(float* input_gradient, float* output_gradient) = 0;
    virtual void backwardParams(float* input, float* output_gradient) = 0;
    virtual void destroyDescriptors() = 0;
    virtual void zeroGradients() = 0;
    
    // Weight access for optimizer
    virtual float* getWeights() = 0;
    virtual float* getWeightGradients() = 0;
    virtual int getWeightSize() = 0;

    protected:
        cudnnHandle_t& cudnn;
};

#endif // LAYER_H 