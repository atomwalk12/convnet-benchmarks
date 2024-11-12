#ifndef LOSS_H
#define LOSS_H

#include <cuda_runtime.h>

class Loss {
public:
    virtual ~Loss() = default;
    virtual float compute(float* prediction, float* target, int size) = 0;
    virtual void backward(float* prediction, float* target, float* gradient, int size) = 0;
};

class MSELoss : public Loss {
public:
    MSELoss() = default;
    ~MSELoss() = default;
    
    float compute(float* prediction, float* target, int size) override;
    void backward(float* prediction, float* target, float* gradient, int size) override;
};

#endif // LOSS_H 