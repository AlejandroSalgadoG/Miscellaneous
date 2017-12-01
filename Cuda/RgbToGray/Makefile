CUDA_INCLUDE_PATH=/opt/cuda/include
OPENCV_INCLUDE_PATH=/usr/include

OPENCV_LIB_PATH=/usr/lib
OPENCV_LIBS=-lopencv_core -lopencv_imgcodecs -lopencv_imgproc

NVCC_OPTS=-Wno-deprecated-gpu-targets

Main: Function.o Main.o
	nvcc Function.o Main.o -o Main -L $(OPENCV_LIB_PATH) $(OPENCV_LIBS) $(NVCC_OPTS)

Main.o: Main.cpp
	g++ -c Main.cpp -I $(CUDA_INCLUDE_PATH) -I $(OPENCV_INCLUDE_PATH)

Function.o: Function.cu Function.h
	nvcc -c Function.cu $(NVCC_OPTS)

clean:
	rm -f *.o Main

exe:
	./Main image.png
