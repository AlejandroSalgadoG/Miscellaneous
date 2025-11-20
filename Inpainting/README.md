# Image Inpainting using Discrete Heat Equation

A Python implementation of image inpainting based on the discrete heat diffusion equation. This project demonstrates how the heat equation can be used to fill in missing or damaged parts of images by propagating information from known regions into unknown regions.

## Table of Contents

- [Overview](#overview)
- [Mathematical Background](#mathematical-background)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Examples](#examples)
- [Files Description](#files-description)
- [How It Works](#how-it-works)
- [Parameters](#parameters)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [References](#references)

## Overview

Image inpainting is the process of reconstructing missing or corrupted parts of an image. This project uses the **heat diffusion equation** to solve the inpainting problem. The heat equation naturally smooths and fills in missing regions by diffusing information from the boundaries (known pixels) into the interior (missing pixels).

The implementation includes:
- **1D heat equation solver** (`main.py`) - Demonstrates basic heat diffusion in one dimension
- **2D heat equation solver** (`main_2d.py`) - Applicable to image inpainting in two dimensions

## Mathematical Background

### The Heat Equation

The heat equation is a partial differential equation (PDE) that describes how heat (or in our case, pixel intensity) diffuses through space over time:

```
∂T/∂t = α∇²T
```

Where:
- `T` is the temperature (or pixel intensity)
- `t` is time
- `α` is the thermal diffusivity coefficient (controls diffusion speed)
- `∇²T` is the Laplacian operator (measures local curvature)

### Discrete Approximation

In the discrete case, we approximate the Laplacian using finite differences:

**1D Case:**
```
∂²T/∂x² ≈ (T[i-1] - 2T[i] + T[i+1]) / dx²
```

**2D Case:**
```
∇²T ≈ (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4T[i,j]) / dx²
```

### Time Integration

The solution evolves using forward Euler integration:

```
T(t + dt) = T(t) + dt × ∂T/∂t
```

## Installation

### Requirements

```bash
pip install numpy matplotlib
```

Or using a requirements.txt file:

```
numpy>=1.20.0
matplotlib>=3.3.0
```

### Clone and Setup

```bash
git clone <repository-url>
cd Inpainting
pip install -r requirements.txt
```

## Usage

### Running the 1D Example

The 1D example demonstrates heat diffusion along a line with fixed boundary conditions:

```bash
python main.py
```

This will display an animated visualization showing how heat diffuses from two boundary temperatures (100°C on the left, 90°C on the right) into a cold interior (0°C).

### Running the 2D Example

The 2D example demonstrates heat diffusion on a grid, which can be used for image inpainting:

```bash
python main_2d.py
```

This will display an animated heatmap showing how temperature diffuses from the boundaries (top=100°C, bottom=0°C, left=100°C, right=0°C) into the interior.

## Implementation Details

### 1D Heat Equation (`main.py`)

**Key Components:**

- **Grid Setup**: Creates a 1D spatial grid with `n` points over length `L`
- **Boundary Conditions**: Fixed temperatures at both ends (`t1=100`, `t2=90`)
- **Time Evolution**: Iterates over time steps, updating temperature distribution
- **Visualization**: Uses LineCollection with color mapping to show temperature distribution

**Key Parameters:**
```python
L = 50          # Domain length
n = 30          # Number of grid points
T0 = 0          # Initial temperature
alpha = 4       # Thermal diffusivity
tf = 60         # Final simulation time
dt = 0.1        # Time step
```

### 2D Heat Equation (`main_2d.py`)

**Key Components:**

- **Grid Setup**: Creates a 2D spatial grid with `n×n` points
- **Boundary Conditions**: Fixed temperatures on all four edges
- **Laplacian Calculation**: Computes discrete Laplacian using 5-point stencil
- **Visualization**: Uses pcolormesh to display 2D temperature field

**Key Parameters:**
```python
L = 200         # Domain size
n = 50          # Grid resolution (50×50)
T0 = 0          # Initial temperature
alpha = 2.5     # Thermal diffusivity
tf = 60         # Final simulation time
dt = 0.1        # Time step
```

## Examples

### 1D Heat Diffusion

The 1D example models a rod with hot ends and a cold middle. Over time, heat flows inward until the system reaches equilibrium.

**Initial State:**
```
|100°C| 0°C  0°C  0°C  0°C  0°C |90°C|
```

**Final State (steady-state):**
```
|100°C| 99°C 98°C 96°C 94°C 92°C |90°C|
```

### 2D Heat Diffusion (Image Inpainting)

The 2D example models a square domain with hot top and left boundaries, and cold bottom and right boundaries. This creates a smooth gradient field.

**Boundary Configuration:**
```
       100°C (top)
100°C |         | 0°C
(left)|    ?    |(right)
       0°C (bottom)
```

The interior values are filled in by solving the heat equation, creating a smooth interpolation.

## Files Description

### `main.py`

1D heat equation solver with:
- Boundary conditions at both ends
- Color-coded visualization showing temperature evolution
- Real-time animation of diffusion process

### `main_2d.py`

2D heat equation solver with:
- Four boundary conditions (one per edge)
- 2D heatmap visualization
- Suitable for image inpainting applications

## How It Works

### The Inpainting Process

1. **Define Known Region**: Pixels outside the damaged area (boundaries)
2. **Define Unknown Region**: Missing or damaged pixels (interior)
3. **Initialize**: Set unknown pixels to an initial value (e.g., 0)
4. **Iterate**: Apply heat diffusion equation repeatedly
5. **Converge**: Continue until the solution stabilizes

### Why Heat Equation for Inpainting?

The heat equation has several properties that make it ideal for inpainting:

1. **Smoothness**: Produces smooth, natural-looking fills
2. **Boundary Respect**: Naturally respects known pixel values at boundaries
3. **Physical Intuition**: Based on well-understood physical process
4. **Mathematical Elegance**: Minimizes the Dirichlet energy (smoothest interpolation)
5. **Simplicity**: Easy to implement and understand

### Connection to Image Processing

In image inpainting:
- **Temperature (T)** → **Pixel Intensity**
- **Thermal Diffusivity (α)** → **Smoothing Parameter**
- **Time Evolution** → **Iterative Filling Process**
- **Steady State** → **Final Inpainted Image**

## Parameters

### Tuning Parameters for Best Results

**Grid Resolution (`n`):**
- Higher values → More detail, slower computation
- Lower values → Faster computation, less detail
- Recommended: 50-200 for images

**Thermal Diffusivity (`alpha`):**
- Higher values → Faster diffusion, more smoothing
- Lower values → Slower diffusion, more detail preservation
- Recommended: 1-5 for images

**Time Step (`dt`):**
- Must satisfy stability condition: `dt ≤ dx²/(4α)` for 2D
- Too large → Numerical instability
- Too small → Slow convergence
- Recommended: 0.01-0.1

**Simulation Time (`tf`):**
- Longer time → Better convergence to steady state
- Shorter time → Partial filling
- Recommended: Run until convergence (when changes become negligible)
