# Fourier Analysis & Visualization

A collection of Python scripts for exploring and visualizing Fourier analysis concepts, including Fourier series, Fast Fourier Transform (FFT), and complex exponentials using interactive matplotlib visualizations.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [FFT Coefficients Analysis](#fft-coefficients-analysis)
  - [Complex Exponentials Explorer](#complex-exponentials-explorer)
  - [Fourier Transform Visualization](#fourier-transform-visualization)
  - [Bézier Curve Fourier Series](#bézier-curve-fourier-series)
  - [Function Plotting](#function-plotting)
- [Mathematical Background](#mathematical-background)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Examples](#examples)
- [Contributing](#contributing)

## Overview

This project provides educational tools for understanding Fourier analysis through interactive visualizations. It demonstrates:

- How signals are decomposed into frequency components using FFT
- The relationship between time and frequency domains
- Complex exponentials and Euler's formula visualization
- Fourier series approximation of arbitrary curves using Bézier paths
- Signal reconstruction from Fourier coefficients

## Features

- **FFT Analysis**: Decompose composite signals and visualize amplitude and phase spectra
- **Interactive Complex Plane**: Navigate through complex exponentials using keyboard controls
- **Frequency Domain Visualization**: See how signals wrap around the complex plane
- **Bézier Curves**: Define complex paths and approximate them using Fourier series
- **Real-time Plotting**: Observe the convergence of Fourier series to target functions

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Fourier
```

2. Install required dependencies:
```bash
pip install numpy matplotlib
```

## Usage

### FFT Coefficients Analysis

**File**: `Coefitients.py`

Demonstrates Fast Fourier Transform on a composite signal made of two cosine waves.

```bash
python Coefitients.py
```

**What it does:**
- Creates a signal composed of two cosine waves (frequencies: 6 Hz and 32 Hz)
- Computes FFT coefficients up to the Nyquist frequency
- Plots amplitude and phase spectra
- Reconstructs individual frequency components and the full signal

**Key Parameters:**
```python
n = 100          # Number of samples
f, f2 = 6, 32    # Frequencies of the two components
a, a2 = 3, 3     # Amplitudes
phi, phi2 = 0, 0 # Phase shifts
```

**Output:**
- Plot 1: Amplitude and phase spectra showing peaks at 6 Hz and 32 Hz
- Plot 2: Reconstructed 6 Hz component vs original signal
- Plot 3: Reconstructed 32 Hz component vs original signal
- Plot 4: Full reconstruction from all Fourier components

### Complex Exponentials Explorer

**File**: `ImExpPlot.py`

Interactive visualization of Euler's formula: \( e^{ix} = \cos(x) + i\sin(x) \)

```bash
python ImExpPlot.py
```

**Controls:**
- **Right Arrow**: Step forward in time
- **Left Arrow**: Step backward in time

**What it does:**
- Plots points on the unit circle in the complex plane
- Shows how complex exponentials rotate around the origin
- Each step advances by π/2 radians (configurable via `time_jump`)

**Key Parameters:**
```python
time_jump = np.pi/2  # Radians per step (π/2 = 90 degrees)
```

### Fourier Transform Visualization

**File**: `Plot.py`

Visualizes how the Fourier transform "wraps" a time-domain signal around the complex plane to extract frequency information.

```bash
python Plot.py
```

**What it does:**
- Creates a cosine wave in the time domain
- Multiplies the signal by a rotating complex exponential
- Plots both the time-domain signal and its path in the complex plane
- Red line shows the wrapped signal trajectory

**Key Parameters:**
```python
f_sin = 2  # Frequency of the input signal
f = 2      # Frequency of the complex exponential (for wrapping)
t = 4.5    # Time duration
```

**Mathematical Concept:**
The Fourier transform integrates the signal multiplied by \( e^{-i2\pi ft} \). When the wrapping frequency matches the signal frequency, the path in the complex plane has a consistent winding direction, resulting in a large magnitude coefficient.

### Bézier Curve Fourier Series

**File**: `Visualize/FourierSeries.py`

The most comprehensive script that demonstrates Fourier series approximation of an arbitrary path defined by Bézier curves.

```bash
cd Visualize
python FourierSeries.py
```

**What it does:**
1. Defines a complex path using 10 connected cubic Bézier curves
2. Treats the path as a complex-valued function: \( f(t) = x(t) + iy(t) \)
3. Computes Fourier series coefficients numerically
4. Reconstructs the function using N terms of the Fourier series
5. Compares original and reconstructed paths

**Key Parameters:**
```python
N = 100      # Number of Fourier terms (harmonics)
dt = 0.005   # Sampling interval for coefficient calculation
```

**Mathematical Concepts:**

**Fourier Coefficient Calculation:**
```
c_n = ∫₀¹ f(t) e^(-i2πnt) dt
```

**Fourier Series Reconstruction:**
```
f̂(t) = Σ c_n e^(i2πnt)  (sum from n=-N to N)
```

**Classes:**
- `Bezier`: Cubic Bézier curve implementation with 4 control points
- `Function`: Container for multiple Bézier curves forming a continuous path

**Output:**
- Three subplots showing real part, imaginary part, and complex plane representation
- Original function vs. Fourier series approximation
- Convergence quality depends on N (number of terms)

### Function Plotting

**File**: `Visualize/PlotFunction.py`

Plots the Bézier-based function in both component and complex forms without Fourier analysis.

```bash
cd Visualize
python PlotFunction.py
```

**Output:**
- Real component vs. parameter t
- Imaginary component vs. parameter t
- Path in the complex plane (x vs. y)

### Bézier Curve Visualization

**File**: `Visualize/PlotBezier.py`

Simple visualization of the Bézier curve path.

```bash
cd Visualize
python PlotBezier.py
```

**Output:**
- Scatter plot showing the Bézier curve path
- 250×250 pixel canvas with y-axis inversion for screen coordinates

## Mathematical Background

### Fourier Series

Any periodic function can be represented as a sum of sines and cosines (or complex exponentials):

```
f(t) = Σ c_n e^(i2πnt)
```

where the coefficients are:

```
c_n = ∫₀¹ f(t) e^(-i2πnt) dt
```

### Fast Fourier Transform (FFT)

The FFT is an efficient algorithm for computing the Discrete Fourier Transform:

```
X[k] = Σ x[n] e^(-i2πkn/N)
```

It reduces computation from O(N²) to O(N log N).

### Euler's Formula

The fundamental relationship between complex exponentials and trigonometry:

```
e^(ix) = cos(x) + i sin(x)
```

This allows representing rotations as complex exponentials and is the foundation of Fourier analysis.

### Bézier Curves

Cubic Bézier curves are defined by 4 control points and parametric interpolation:

```
B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
```

where t ∈ [0,1]. Multiple curves can be chained to form complex paths.

## File Structure

```
Fourier/
├── Coefitients.py           # FFT analysis of composite signals
├── ImExpPlot.py             # Interactive complex exponential visualization
├── Plot.py                  # Fourier transform wrapping visualization
├── Visualize/
│   ├── FourierSeries.py     # Bézier curve Fourier series approximation
│   ├── PlotBezier.py        # Bézier curve plotter
│   ├── PlotFunction.py      # Function visualization (real, imag, complex)
│   └── trevol.svg           # SVG resource (if used for curve definition)
└── README.md                # This file
```

## Dependencies

- **numpy**: Numerical computing and array operations
- **matplotlib**: Plotting and visualization

Install with:
```bash
pip install numpy matplotlib
```

**Version Requirements:**
- numpy >= 1.20.0
- matplotlib >= 3.3.0

## Examples

### Example 1: Analyzing a Simple Signal

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a signal
t = np.linspace(0, 1, 100)
signal = 3*np.cos(2*np.pi*5*t) + 2*np.sin(2*np.pi*10*t)

# Compute FFT
fft_coef = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), t[1]-t[0])

# Plot
plt.plot(frequencies[:50], np.abs(fft_coef)[:50])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()
```

### Example 2: Creating a Simple Bézier Curve

```python
from Visualize.FourierSeries import Bezier
import matplotlib.pyplot as plt
import numpy as np

# Define a Bézier curve
curve = Bezier(
    p1=(0, 0),      # Start point
    d1=(10, 20),    # Control point 1 offset
    d2=(30, 20),    # Control point 2 offset
    d3=(40, 0)      # End point offset
)

# Plot the curve
for t in np.linspace(0, 1, 100):
    x, y = curve.calculate(t)
    plt.scatter(x, y, s=1, c='blue')

plt.show()
```
