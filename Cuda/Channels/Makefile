CUDA_INCLUDE_PATH=/opt/cuda/include
OPENCV_INCLUDE_PATH=/usr/include

OPENCV_LIB_PATH=/usr/lib
OPENCV_LIBS=-lopencv_core -lopencv_imgcodecs -lopencv_imgproc

NVCC_OPTS=-Wno-deprecated-gpu-targets

Main: My_cuda_kernels.o Main.o
	nvcc My_cuda_kernels.o Main.o -o Main -L $(OPENCV_LIB_PATH) $(OPENCV_LIBS) $(NVCC_OPTS)

Main.o: Main.cpp
	g++ -c Main.cpp -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

My_cuda_kernels.o: My_cuda_kernels.cu My_cuda_kernels.h
	nvcc -c My_cuda_kernels.cu $(NVCC_OPTS)

clean:
	rm -f *.o Main Image*

exe:
	./Main image.png
