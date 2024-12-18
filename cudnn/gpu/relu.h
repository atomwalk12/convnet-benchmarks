#ifndef RELU_GPU_H
#define RELU_GPU_H

class ReLU_GPU {
    private:
        float* forward_input;
        int sz_out;
        static const int block_size = 256;
        int n_blocks;

        
    public:
        ReLU_GPU(int _sz_out);
        void forward(float* _inp, float* _out);
        void backward(float* gradient_out, float* gradient_in);

        size_t getSzOut() const { return sz_out; }

};

#endif
