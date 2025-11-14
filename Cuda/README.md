# CUDA Image Processing Suite

A collection of GPU-accelerated image processing programs written in CUDA C/C++ and OpenCV. This project demonstrates parallel computing techniques for common image manipulation tasks using NVIDIA CUDA.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Programs](#programs)
  - [RGB to Grayscale](#1-rgb-to-grayscale)
  - [Image Blur](#2-image-blur)
  - [Channel Separation](#3-channel-separation)
- [Building](#building)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Performance Considerations](#performance-considerations)
- [License](#license)

## Overview

This repository contains three CUDA-based image processing applications that leverage GPU parallelism to perform efficient image transformations. Each program is self-contained with its own directory, Makefile, and CUDA kernels.

## Features

- **GPU-Accelerated Processing**: Utilizes NVIDIA CUDA for parallel computation
- **OpenCV Integration**: Seamless image I/O using OpenCV library
- **Multiple Algorithms**: RGB to grayscale conversion, blur filtering, and channel separation
- **Optimized Kernels**: Custom CUDA kernels designed for efficient GPU execution

## Requirements

### Software Dependencies

- **CUDA Toolkit** (version 8.0 or later recommended)
  - NVCC compiler
  - CUDA runtime libraries
- **OpenCV** (version 3.0 or later)
  - opencv_core
  - opencv_imgcodecs
  - opencv_imgproc
- **G++** compiler (for C++ host code)
- **Make** build system

### Hardware Requirements

- NVIDIA GPU with CUDA support (Compute Capability 2.0 or higher)
- Sufficient GPU memory for image processing (depends on image size)

### System

The project has been developed and tested on Linux systems. Path configurations may need adjustment for other platforms.

## Installation

### 1. Install CUDA Toolkit

```bash
# Download from NVIDIA's official website
# https://developer.nvidia.com/cuda-downloads
# Follow installation instructions for your platform
```

### 2. Install OpenCV

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install libopencv-dev

# Fedora
sudo dnf install opencv-devel

# Or build from source
# https://opencv.org/releases/
```

### 3. Clone the Repository

```bash
git clone <repository-url>
cd Cuda
```

### 4. Configure Paths

Edit the Makefiles in each subdirectory to match your system paths:

```makefile
CUDA_INCLUDE_PATH=/opt/cuda/include      # Your CUDA include path
OPENCV_INCLUDE_PATH=/usr/include          # Your OpenCV include path
OPENCV_LIB_PATH=/usr/lib                  # Your OpenCV library path
```

## Project Structure

```
Cuda/
├── README.md
├── RgbToGray/
│   ├── Main.cpp              # Host code for RGB to grayscale conversion
│   ├── Function.cu           # CUDA kernel implementation
│   ├── Function.h            # Header file
│   └── Makefile              # Build configuration
├── Blur/
│   ├── Main.cpp              # Host code for image blurring
│   ├── My_cuda_kernels.cu    # CUDA kernel implementations
│   ├── My_cuda_kernels.h     # Header file
│   └── Makefile              # Build configuration
└── Channels/
    ├── Main.cpp              # Host code for channel separation
    ├── My_cuda_kernels.cu    # CUDA kernel implementation
    ├── My_cuda_kernels.h     # Header file
    └── Makefile              # Build configuration
```

## Programs

### 1. RGB to Grayscale

Converts color images to grayscale using the standard luminosity formula.

**Algorithm**: `I = 0.299 × R + 0.587 × G + 0.114 × B`

**Location**: `RgbToGray/`

**Input**: RGB/RGBA color image
**Output**: Grayscale image

**CUDA Kernel**: `rgba_to_greyscale()`
- Converts each pixel using weighted color channel combination
- Parallel execution with one thread per pixel

### 2. Image Blur

Applies a blur filter to images using a 3×3 convolution kernel.

**Location**: `Blur/`

**Input**: RGB/RGBA color image
**Output**: Blurred RGB/RGBA image

**Blur Filter**:
```
[0.0  0.2  0.0]
[0.2  0.2  0.2]
[0.0  0.2  0.0]
```

**CUDA Kernels**:
- `split_channels()`: Separates RGB channels into individual arrays
- `blur_channel()`: Applies convolution filter to each channel independently
- `combine_channels()`: Merges processed channels back into RGB image

### 3. Channel Separation

Extracts and isolates individual RGB color channels from an image.

**Location**: `Channels/`

**Input**: RGB/RGBA color image
**Output**: Three separate images (ImageR.png, ImageG.png, ImageB.png)
- Red channel: Only red component preserved
- Green channel: Only green component preserved
- Blue channel: Only blue component preserved

**CUDA Kernel**: `split_channels()`
- Creates three separate output images
- Zeros out non-relevant color channels for each output

## Building

Each program can be built independently using its Makefile.

### Build Individual Programs

```bash
# RGB to Grayscale
cd RgbToGray
make

# Image Blur
cd Blur
make

# Channel Separation
cd Channels
make
```

### Clean Build Artifacts

```bash
# In any program directory
make clean
```

## Usage

### RGB to Grayscale

```bash
cd RgbToGray
./Main <input_image> <output_image>

# Example
./Main input.png output_gray.png
```

### Image Blur

```bash
cd Blur
./Main <input_image> <output_image>

# Example
./Main input.png output_blurred.png
```

### Channel Separation

```bash
cd Channels
./Main <input_image>

# Example
./Main input.png

# This will generate three output files:
# - ImageR.png (red channel)
# - ImageG.png (green channel)
# - ImageB.png (blue channel)
```

### Supported Image Formats

The programs support common image formats through OpenCV:
- PNG
- JPEG/JPG
- BMP
- TIFF
- And others supported by OpenCV

## Technical Details

### CUDA Configuration

All programs use the following CUDA execution configuration:

```cpp
dim3 blockSize(360, 1, 1);   // 360 blocks
dim3 threadSize(480, 1, 1);  // 480 threads per block
```

**Note**: This configuration is hardcoded for specific image dimensions (360×480). For production use, these should be calculated dynamically based on input image size.

### Memory Management

Each program follows this workflow:

1. **Host-side**:
   - Load image using OpenCV
   - Convert to RGBA format
   - Allocate device memory

2. **Device-side**:
   - Transfer image data to GPU
   - Execute CUDA kernels
   - Transfer results back to host

3. **Cleanup**:
   - Free device memory
   - Save output image

### Thread Indexing

The programs use 2D thread indexing for pixel operations:

```cpp
int row = threadIdx.x;       // Thread index within block
int col = blockIdx.x;        // Block index
int idx = col + row * 360;   // Linear index for 1D array
```

### Data Types

- `uchar4`: CUDA structure for 4-channel unsigned char (RGBA)
  - `.x` = Red channel
  - `.y` = Green channel
  - `.z` = Blue channel
  - `.w` = Alpha channel
- `unsigned char`: Single-channel grayscale data

## Performance Considerations

### Optimization Opportunities

1. **Dynamic Grid Configuration**: Currently hardcoded for 360×480 images
   ```cpp
   // Should be calculated as:
   dim3 blockSize((numCols + 15) / 16, (numRows + 15) / 16);
   dim3 threadSize(16, 16);
   ```

2. **Boundary Checking**: Blur kernel has potential out-of-bounds access
   - Current conditions check against 0 but don't prevent negative indices

3. **Shared Memory**: Could utilize shared memory for blur convolution to reduce global memory access

4. **Memory Coalescing**: Thread access patterns could be optimized for better memory throughput

5. **Error Checking**: Add CUDA error checking after kernel launches and memory operations
   ```cpp
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
       // Handle error
   }
   ```

### Benchmark Guidelines

For accurate performance measurements:
- Use `cudaEvent_t` for timing GPU operations
- Compare against CPU implementations
- Test with various image sizes
- Profile using NVIDIA Nsight or nvprof
