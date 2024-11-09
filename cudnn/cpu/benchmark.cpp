#include <dnnl.hpp>
#include <iostream>
#include <vector>

using namespace dnnl;

int main() {
    try {
        // Create execution engine (CPU)
        engine eng(engine::kind::cpu, 0);
        
        // Create stream for the engine
        stream strm(eng);
        
        // Print some info
        std::cout << "oneDNN version: " << dnnl_version() << std::endl;
        std::cout << "Engine created successfully" << std::endl;
        
        // Try to create a simple memory descriptor
        memory::dims dims = {1, 3, 224, 224};  // Same as our AlexNet input
        auto md = memory::desc(dims, memory::data_type::f32, memory::format_tag::nchw);
        
        std::cout << "Memory descriptor created successfully" << std::endl;
        
    } catch (dnnl::error& e) {
        std::cerr << "oneDNN error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}