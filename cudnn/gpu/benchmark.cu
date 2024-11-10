#include <stdio.h>
#include "alexnet.h"
#include <chrono>

// Benchmark parameters
const int NUM_ITERATIONS = 100;
const int WARMUP_ITERATIONS = 1;

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
    // ================================
    // =====      Input data      =====
    // ================================
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
    AlexNet model(cudnn, batch_size, num_classes);

    // Create dummy gradient for backward pass
    float* output_gradient = model.createDummyGradient(output_data);
    float* input_gradient;
    cudaMallocManaged(&input_gradient, batch_size*channels*height*width*sizeof(float));
    cudaDeviceSynchronize();

    // ================================
    // =====      Warmup run      =====
    // ================================
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        model.forward(input_data, output_data);
        model.backwardInput(input_gradient, output_gradient);
        model.backwardParams(input_data, output_gradient);
    }
    cudaDeviceSynchronize();

    // ================================
    // =====      Timing run      ===== 
    // ================================
    // =====     Forward pass     =====
    // ================================
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        model.forward(input_data, output_data);
    }
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();

    double total_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    double average_milliseconds = (total_microseconds / 1000.0) / NUM_ITERATIONS;
    double total_time = average_milliseconds;
    printf("Average forward pass time: %f ms\n", average_milliseconds);

    // ================================
    // ===== Backward input pass ======
    // ================================
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        model.backwardInput(input_gradient, output_gradient);
    }
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    
    total_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    average_milliseconds = (total_microseconds / 1000.0) / NUM_ITERATIONS;
    total_time += average_milliseconds;
    printf("Average backward input pass time: %f ms\n", average_milliseconds);
    
    // ================================
    // ===== Backward params pass =====
    // ================================
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        model.backwardParams(input_data, output_gradient);
    }
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    
    total_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    average_milliseconds = (total_microseconds / 1000.0) / NUM_ITERATIONS;
    total_time += average_milliseconds;
    printf("Average backward params pass time: %f ms\n", average_milliseconds);
    printf("Total time: %f ms\n", total_time);

    // Additional cleanup
    cudaFree(input_gradient);
    cudaFree(output_gradient);

    // Cleanup
    cudaFree(input_data);
    cudaFree(output_data);
    cudnnDestroy(cudnn);

    return 0;
}

