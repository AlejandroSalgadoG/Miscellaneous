#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.hpp>

#include <fstream>
#include <iostream>
#include <string>

cl::Platform get_platform(){
    std::vector< cl::Platform > platform_list;
    cl::Platform::get(&platform_list);
    return platform_list.front();
}

cl::Device get_device( cl::Platform platform ){
    std::vector<cl::Device> devices;
    platform.getDevices( CL_DEVICE_TYPE_GPU, &devices );
    return devices.front();
}

cl::Program read_program( cl::Context context ){
    std::ifstream file("Kernel.cl");
    std::string code( std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source( 1, std::make_pair(code.c_str(), code.length()+1) );
    cl::Program program(context, source);
    return program;
}

int main(void) {
    cl::Platform platform = get_platform();
    cl::Device device = get_device( platform );
    cl::Context context( device );
    cl::Program program = read_program( context );

    program.build("-cl-std=CL1.2");

    size_t msg_length = 7;
    char *h_buf = new char[msg_length];
    cl::Buffer d_buf( context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, msg_length, h_buf );

    cl::Kernel kernel(program, "hello");
    kernel.setArg(0, d_buf);

    cl::CommandQueue queue( context, device, 0 );

    queue.enqueueNDRangeKernel( kernel, cl::NullRange, cl::NDRange(msg_length), cl::NDRange(1, 1) );
    queue.enqueueReadBuffer( d_buf, CL_TRUE, 0, msg_length, h_buf);

    std::cout << h_buf;

    return EXIT_SUCCESS;
}
