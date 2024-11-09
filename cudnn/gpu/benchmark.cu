#include <stdio.h>
#include "alexnet.h"
#include <chrono>

void checkCUDNN(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("cuDNN Error: %s\n", cudnnGetErrorString(status));
        exit(1);
    }
}

int main() {
    std::chrono::steady_clock::time_point begin, end;

    // Initialize CUDNN
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // The data format is NCHW
    const int batch_size = 64;    // N
    const int channels = 3;       // C
    const int height = 224;       // H
    const int width = 224;        // W
    const int num_classes = 1000;

    // These vectors will be initialized with random values
    float *input_data, *output_data;
    cudaMallocManaged(&input_data, batch_size*channels*height*width*sizeof(float));
    cudaMallocManaged(&output_data, batch_size*num_classes*sizeof(float));

    for (int i = 0; i < batch_size * channels * height * width; i++) {
        input_data[i] = (float)rand() / RAND_MAX;
    }

    for (int i = 0; i < batch_size * num_classes; i++) {
        output_data[i] = 0.0f;
    }

    // Create and initialize the AlexNet model
    AlexNet model(cudnn, batch_size);

    // Cleanup
    cudaFree(input_data);
    cudaFree(output_data);

    return 0;
}

