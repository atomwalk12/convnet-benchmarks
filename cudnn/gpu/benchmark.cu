#include <stdio.h>
#include "alexnet.h"
#include <chrono>
#include "loss.h"

// Benchmark parameters
const int NUM_ITERATIONS = 100;
const int WARMUP_ITERATIONS = 300;

// // Network parameters
// const int BATCH_SIZE = 3;
// const int NUM_CLASSES = 3;

// // Modified for AlexNet input specifications
// const int IN_CHANNELS = 3;  // RGB input
// const int INPUT_HEIGHT = 224;
// const int INPUT_WIDTH = 224;

// // Modified for AlexNet first conv layer
// const int CONV_OUT_CHANNELS = 96;  // AlexNet's first layer filter count
// const int CONV_KERNEL_SIZE = 11;   // 11x11 kernel
// const int CONV_STRIDE = 4;         // Stride of 4
// const int CONV_PADDING = 0;        // No padding

// Network parameters
const int BATCH_SIZE = 1;
const int NUM_CLASSES = 3;

// Modified for AlexNet input specifications
const int IN_CHANNELS = 1;
const int INPUT_HEIGHT = 2;
const int INPUT_WIDTH = 3;

// Modified for AlexNet first conv layer
const int CONV_OUT_CHANNELS = 2;  // AlexNet's first layer filter count
const int CONV_KERNEL_SIZE = 2;   // 11x11 kernel
const int CONV_STRIDE = 1;         // Stride of 4
const int CONV_PADDING = 0;        // No padding

// Define sizes in terms of number of elements
const int INPUT_SIZE = BATCH_SIZE * IN_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT;
const int OUTPUT_SIZE = BATCH_SIZE * NUM_CLASSES;
const int INPUT_GRADIENT_SIZE = BATCH_SIZE * IN_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT;

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

    // Create network
    Network model(cudnn, BATCH_SIZE, NUM_CLASSES);
    
    // Add layers with AlexNet parameters
    model.addConvLayer(
        INPUT_WIDTH,
        INPUT_HEIGHT,
        IN_CHANNELS,
        CONV_OUT_CHANNELS,
        CONV_KERNEL_SIZE,
        CONV_STRIDE,
        CONV_PADDING
    );
    
    // Note: You might need to adjust the FC layer input size based on the conv output
    const int conv_output_height = (INPUT_HEIGHT - CONV_KERNEL_SIZE + 2 * CONV_PADDING) / CONV_STRIDE + 1;
    const int conv_output_width = (INPUT_WIDTH - CONV_KERNEL_SIZE + 2 * CONV_PADDING) / CONV_STRIDE + 1;
    
    model.addFCLayer(
        CONV_OUT_CHANNELS * conv_output_width * conv_output_height,
        NUM_CLASSES
    );

    // These vectors will be initialized with random values
    // ================================
    // =====      Input data      =====
    // ================================
    float *input_data, *output_data;
    
    cudaMallocManaged(&input_data, INPUT_SIZE * sizeof(float));
    cudaMallocManaged(&output_data, OUTPUT_SIZE * sizeof(float));

        // Create input data with explicit layout
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < IN_CHANNELS; c++) {
            for (int h = 0; h < INPUT_HEIGHT; h++) {
                for (int w = 0; w < INPUT_WIDTH; w++) {
                    int idx = ((b * IN_CHANNELS + c) * INPUT_HEIGHT + h) * INPUT_WIDTH + w;
                    input_data[idx] = (float)rand() / RAND_MAX;
                }
            }
        }
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_data[i] = 0.0f;
    }

    // Create dummy gradient for backward pass
    float* output_gradient = model.createDummyGradient(output_data);
    float* input_gradient;
    cudaMallocManaged(&input_gradient, INPUT_GRADIENT_SIZE * sizeof(float));
    cudaDeviceSynchronize();

    MSELoss loss;

    // ================================
    // =====      Warmup run      =====
    // ================================
    // MSELoss loss;
    
    // Target data still needs size, but we get it from the model
    float* target_data;
    cudaMallocManaged(&target_data, OUTPUT_SIZE * sizeof(float));

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        target_data[i] = 1.0f;
    }
    
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        model.zeroGradients();
        
        // Forward pass
        model.forward(input_data, output_data);
        
        // Compute loss and gradients
        float loss_value = loss.compute(output_data, target_data, OUTPUT_SIZE);
        loss.backward(output_data, target_data, output_gradient, OUTPUT_SIZE);
        
        // Backward pass
        model.backwardInput(input_gradient, output_gradient);
        model.backwardParams(input_data, output_gradient);
        
        // Update weights
        model.updateWeights(0.001f);
        
        
        printf("Iteration %d, Loss: %f\n", i, loss_value);
    }

    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(target_data);
    
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

