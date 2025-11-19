# System Identification

A Python implementation for system identification using various estimation techniques including Kalman Filtering, Least Squares methods, and Time Series analysis.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Modules](#modules)
  - [Kalman Filter](#kalman-filter)
  - [Least Squares](#least-squares)
  - [Time Series](#time-series)
- [Usage Examples](#usage-examples)
- [Mathematical Background](#mathematical-background)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## Overview

This repository provides implementations of fundamental system identification algorithms used in control theory, signal processing, and time series analysis. System identification is the art and science of building mathematical models of dynamic systems from observed input-output data.

## Features

- **Kalman Filtering**: State estimation and parameter identification for linear dynamical systems
- **Least Squares Method**: System parameter estimation from input-output data
- **ARMA Models**: Time series generation and analysis with various noise distributions
- **Visualization Tools**: Built-in plotting functions for analysis and validation
- **Noise Modeling**: Support for Gaussian and Student-t distributed noise

## Project Structure

```
SysIdentification/
│
├── Kalman/
│   ├── AwesomeKalman.py           # Standard Kalman filter implementation
│   └── AwesomeKalmanEstimation.py # Kalman filter for parameter estimation
│
├── LeastSquares/
│   ├── hidden.py                   # Example hidden system
│   ├── linear.py                   # Linear algebra utilities
│   ├── method.py                   # Least squares algorithm
│   └── main.py                     # Demonstration script
│
└── TimeSeries/
    └── Arma_application.py         # ARMA model implementation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SysIdentification.git
cd SysIdentification
```

2. Install required dependencies:
```bash
pip install numpy matplotlib
```

## Modules

### Kalman Filter

The Kalman filter is an optimal recursive algorithm for estimating the state of a linear dynamic system from noisy measurements.

#### Files

##### `AwesomeKalman.py`

Standard Kalman filter implementation for state estimation.

**Key Functions:**
- `simulate(A, x_0, Xi, T)`: Simulates a linear dynamical system with process noise
- `noisy_observer(H, x, Zeta, T)`: Generates noisy observations from true states
- `kalman_filter(F, H, R, Q, z_0, y, T)`: Implements the Kalman filter algorithm
- `plot_simulations(y_real, y, y_kalman, var, T)`: Visualizes ideal, noisy, and filtered trajectories
- `plot_variance(sigma, T)`: Plots the trace of estimation covariance over time

**System Model:**
```
x[k+1] = A * x[k] + ξ[k]     (State equation)
y[k]   = H * x[k] + ζ[k]     (Observation equation)
```

Where:
- `x[k]`: State vector at time k
- `y[k]`: Observation vector at time k
- `A`: State transition matrix
- `H`: Observation matrix
- `ξ[k]`: Process noise ~ N(0, Q)
- `ζ[k]`: Measurement noise ~ N(0, R)

##### `AwesomeKalmanEstimation.py`

Extended Kalman filter for parameter estimation in addition to state estimation.

**Key Features:**
- Augmented state vector containing both system states and unknown parameters
- Time-varying state transition matrix
- Parameter convergence visualization

**Use Case:** Estimating unknown coefficients (a₁, a₂, a₃) in a dynamical system while simultaneously tracking the system state.

### Least Squares

The least squares method is used to identify system parameters by minimizing the sum of squared residuals between observed and predicted outputs.

#### Files

##### `linear.py`

Provides fundamental linear algebra operations.

**Functions:**
- `vector(values)`: Converts a list to a column vector
- `multiply(A, B)`: Matrix multiplication wrapper
- `invert(A)`: Matrix inversion wrapper

##### `method.py`

Core least squares implementation.

**Functions:**
- `least_squares(inputs, outputs)`: Computes the least squares estimate
- `calc_v(inputs, outputs)`: Calculates the V matrix (∑ x·yᵀ)
- `calc_g(inputs)`: Calculates the G matrix (∑ x·xᵀ)

**Algorithm:**
```
Given: y = A·x + noise
Estimate: Â = (∑ y·xᵀ) · (∑ x·xᵀ)⁻¹
```

##### `hidden.py`

Demonstration system with unknown parameters.

**System Definition:**
```python
A = [[1, 2],
     [3, 4]]
y = A·x + N(0, 0.1)
```

##### `main.py`

Example script demonstrating system identification using least squares to recover the hidden matrix A.

### Time Series

Time series analysis tools for generating and analyzing ARMA models.

#### Files

##### `Arma_application.py`

Implements ARMA (AutoRegressive Moving Average) model for time series generation.

**Model Equation:**
```
r[t] = φ₀ + φ₁·r[t-1] + a[t] - θ₁·a[t-1] - θ₂·a[t-2]
```

Where:
- `r[t]`: Time series value at time t
- `φ`: AR (AutoRegressive) parameters
- `θ`: MA (Moving Average) parameters
- `a[t]`: White noise process

**Key Functions:**
- `arma_series(T, r_0, phi, theta, distribution, params)`: Generates ARMA time series
- `norm_numbers(params, shape)`: Gaussian noise generator
- `t_numbers(params, shape)`: Student-t noise generator

**Features:**
- Support for multiple noise distributions
- Comparison between Gaussian and heavy-tailed (Student-t) noise
- Statistical analysis (mean and variance computation)

## Usage Examples

### Example 1: Kalman Filter for State Estimation

```python
import numpy as np
from Kalman.AwesomeKalman import *

# Define system parameters
T = 20
x_0 = np.array([0, 0, 2])
A = np.array([[1, 1, 0.5], [0, 1, 1], [0, 0, 1]])
H = np.array([[1, 0, 0], [0, 1, 0]])

# Define noise covariances
Q = np.array([[5, 0, 0], [0, 1, 0], [0, 0, 0.1]])**2
R = np.array([[35, 0], [0, 5]])**2

# Generate noise
np.random.seed(37756813)
Xi = get_zero_mean_normal_vars(Q, T+1)
Zeta = get_zero_mean_normal_vars(R, T+1)

# Simulate system
x = simulate(A, x_0, Xi, T)
y = noisy_observer(H, x, Zeta, T)

# Apply Kalman filter
x_kalman, s_kalman = kalman_filter(A, H, R, Q, x_0, y, T)

# Visualize results
y_kalman = perfect_observer(H, x_kalman, T)
plot_simulations(y_ideal, y, y_kalman, 0, T)
```

### Example 2: Least Squares System Identification

```python
from LeastSquares.hidden import secret_system
from LeastSquares.method import least_squares
from LeastSquares.linear import vector

# Create input data
x1 = vector([1, 3])
x2 = vector([2, 4])

# Observe outputs from hidden system
y1 = secret_system(x1)
y2 = secret_system(x2)

inputs = [x1, x2]
outputs = [y1, y2]

# Estimate system matrix
A_hat = least_squares(inputs, outputs)
print("Estimated system matrix:")
print(A_hat)
```

### Example 3: ARMA Time Series Generation

```python
from TimeSeries.Arma_application import *

# Parameters
T = 200
r_0 = 5.6
phi = [15.5, 0.779]      # AR parameters
theta = [-0.0445, 0.382] # MA parameters

# Generate time series with different noise types
r_norm = arma_series(T, r_0, phi, theta, norm_numbers, [0, 1])
r_stud = arma_series(T, r_0, phi, theta, t_numbers, [2])

# Compare statistics
print(f"Gaussian noise - Mean: {np.mean(r_norm):.2f}, Var: {np.var(r_norm):.2f}")
print(f"Student-t noise - Mean: {np.mean(r_stud):.2f}, Var: {np.var(r_stud):.2f}")

# Visualize
plt.plot(r_norm, label='Gaussian')
plt.plot(r_stud, label='Student-t')
plt.legend()
plt.show()
```

## Mathematical Background

### Kalman Filter

The Kalman filter operates in two stages:

**Prediction Step:**
```
x̂[k|k-1] = F·x̂[k-1|k-1]
P[k|k-1] = F·P[k-1|k-1]·Fᵀ + Q
```

**Update Step:**
```
K[k] = P[k|k-1]·Hᵀ·(H·P[k|k-1]·Hᵀ + R)⁻¹
x̂[k|k] = x̂[k|k-1] + K[k]·(y[k] - H·x̂[k|k-1])
P[k|k] = (I - K[k]·H)·P[k|k-1]
```

Where:
- `K[k]`: Kalman gain
- `P[k]`: Estimation error covariance
- `x̂[k|k]`: State estimate at time k given observations up to time k

### Least Squares

For a linear system `y = A·x + ε`, the least squares estimate minimizes:
```
J = Σ ||y[i] - A·x[i]||²
```

The solution is:
```
Â = (Σ y[i]·x[i]ᵀ)·(Σ x[i]·x[i]ᵀ)⁻¹
```

### ARMA Models

ARMA(p,q) combines:
- **AR(p)**: AutoRegressive of order p
- **MA(q)**: Moving Average of order q

General form:
```
y[t] = φ₀ + Σφᵢ·y[t-i] + Σθⱼ·ε[t-j] + ε[t]
```

## Dependencies

- **NumPy** (≥ 1.19.0): Numerical computing
- **Matplotlib** (≥ 3.3.0): Visualization and plotting
- **Python** (≥ 3.7)

Install all dependencies:
```bash
pip install numpy matplotlib
```

## Key Concepts

### State Estimation vs Parameter Estimation

- **State Estimation**: Tracking the evolving state of a system (e.g., position, velocity)
- **Parameter Estimation**: Identifying unknown system parameters (e.g., mass, damping coefficient)

### Process Noise vs Measurement Noise

- **Process Noise (Q)**: Uncertainty in the system model
- **Measurement Noise (R)**: Uncertainty in the observations

### Model Selection

- Use **Kalman Filter** when: You have a dynamic system model and need real-time state estimation
- Use **Least Squares** when: You need to identify static system parameters from batch data
- Use **ARMA** when: Working with time series data and need to model temporal dependencies

## Performance Considerations

1. **Kalman Filter**: Computational complexity is O(n³) per time step, where n is the state dimension
2. **Least Squares**: Matrix inversion is O(m³), where m is the number of parameters
3. **ARMA**: Linear time complexity in the length of the time series
