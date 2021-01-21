#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.hpp>

#include <fstream>
#include <iostream>
#include <string>

int main(void) {
    cl_int err;

    std::vector< cl::Platform > platform_list;
    cl::Platform::get(&platform_list);
    cl::Platform platform = platform_list.front();

    cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform(), 0 };
    cl::Context context( CL_DEVICE_TYPE_GPU, cprops, NULL, NULL, &err );

    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::Device device = devices.front();

    size_t msg_length = 7;
    char * h_buf = new char[msg_length];
    cl::Buffer d_buf( context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, msg_length, h_buf, &err);

    std::ifstream file("Kernel.cl");
    std::string prog( std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source( 1, std::make_pair(prog.c_str(), prog.length()+1) );

    cl::Program program(context, source);
    err = program.build("-cl-std=CL1.2");

    cl::Kernel kernel(program, "hello", &err);
    err = kernel.setArg(0, d_buf);

    cl::CommandQueue queue(context, device, 0, &err);

    cl::Event event;
    err = queue.enqueueNDRangeKernel( kernel, cl::NullRange, cl::NDRange(msg_length), cl::NDRange(1, 1), NULL, &event);
    event.wait();
    err = queue.enqueueReadBuffer( d_buf, CL_TRUE, 0, msg_length, h_buf);
    std::cout << h_buf;

    return EXIT_SUCCESS;
}
