#include <stdio.h>
#include "alexnet.h"
#include <chrono>
#include "loss.h"

// Benchmark parameters
const int NUM_ITERATIONS = 100;
const int WARMUP_ITERATIONS = 10;

// Network parameters
const int BATCH_SIZE = 3;
const int NUM_CLASSES = 3;

const int IN_CHANNELS = 1;
const int INPUT_HEIGHT = 3;
const int INPUT_WIDTH = 3;

const int CONV_OUT_CHANNELS = 3; // Filter count
const int CONV_KERNEL_SIZE = 3; // Spatial extent
const int CONV_STRIDE = 1;
const int CONV_PADDING = 1;

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
    
    // Add layers with custom parameters
    model.addConvLayer(
        INPUT_WIDTH,
        INPUT_HEIGHT,
        IN_CHANNELS,
        CONV_OUT_CHANNELS,
        CONV_KERNEL_SIZE,
        CONV_STRIDE,
        CONV_PADDING
    );
    
    model.addFCLayer(
        CONV_OUT_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT,
        NUM_CLASSES
    );

    // These vectors will be initialized with random values
    // ================================
    // =====      Input data      =====
    // ================================
    float *input_data, *output_data;
    
    cudaMallocManaged(&input_data, INPUT_SIZE * sizeof(float));
    cudaMallocManaged(&output_data, OUTPUT_SIZE * sizeof(float));

    for (int i = 0; i < INPUT_SIZE; i++) {
        input_data[i] = (float)rand() / RAND_MAX;
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

