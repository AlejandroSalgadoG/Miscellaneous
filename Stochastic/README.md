# Stochastic Processes & Option Pricing

A Python implementation of stochastic processes simulation, numerical methods for stochastic differential equations (SDEs), parameter estimation, and option pricing methods. This project demonstrates the application of stochastic calculus to financial modeling and derivatives pricing.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Modules](#modules)
  - [Brownian Motion](#brownian-motion)
  - [Numerical Methods](#numerical-methods)
  - [Parameter Estimation](#parameter-estimation)
  - [Option Valuation](#option-valuation)
  - [Utility Functions](#utility-functions)
  - [Statistical Testing](#statistical-testing)
- [Usage Examples](#usage-examples)
- [Main Analysis Pipeline](#main-analysis-pipeline)
- [Mathematical Background](#mathematical-background)
- [Requirements](#requirements)
- [Data](#data)

## Overview

This project implements various stochastic processes and numerical methods for solving stochastic differential equations, with a focus on financial applications. It includes:

- Multiple types of Brownian motion simulations
- Numerical solvers for SDEs (Euler, Milstein, Exact solutions)
- Parameter estimation techniques for stochastic processes
- Four different option pricing methods (Black-Scholes, Monte Carlo, Binomial Tree, Finite Differences)
- Real-world stock data analysis and normality testing

## Features

- **Brownian Motion Simulation**: Standard, bridge, drift, and geometric Brownian motion
- **SDE Numerical Methods**: Homogeneous equation solver, Euler method, Milstein method
- **Mean Reversion Models**: CEV (Constant Elasticity of Variance) process simulation
- **Parameter Estimation**: Maximum likelihood estimation for drift and volatility parameters
- **Option Pricing**: Multiple methods with sensitivity analysis
- **Statistical Testing**: Normality tests for returns (Jarque-Bera, Shapiro-Wilk, Lilliefors)
- **Visualization**: Plot generation for simulations and predictions

## Installation

### Prerequisites

- Python 3.7+
- R (for statistical testing)

### Python Dependencies

```bash
pip install numpy scipy matplotlib pandas
```

### R Dependencies

```r
install.packages(c("BatchGetSymbols", "nortest", "tseries", "ggpubr"))
```

## Project Structure

```
Stochastic/
├── Brownian.py          # Brownian motion simulations
├── Numerical.py         # Numerical methods for SDEs
├── Estimation.py        # Parameter estimation
├── OptionValuation.py   # Option pricing methods
├── Functions.py         # Utility functions
├── Final.py             # Main analysis pipeline
├── StatTest.r           # Statistical normality tests
├── AABA_2019.csv        # Stock price data
└── README.md            # This file
```

## Modules

### Brownian Motion

**File**: `Brownian.py`

Implements various types of Brownian motion processes:

#### Functions

- **`standard_brownian(n, T, k)`**: Generates standard Brownian motion paths
  - `n`: Number of time steps
  - `T`: Time horizon
  - `k`: Number of simulated paths
  - Returns: (bt, b1, t) - paths, increments, time array

- **`bridge_brownian(n, k)`**: Generates Brownian bridge paths
  - Conditioned to return to 0 at time T=1
  - Returns: (wt, t) - bridge paths, time array

- **`drift_brownian(n, T, k, u, s)`**: Brownian motion with drift
  - `u`: Drift parameter (μ)
  - `s`: Volatility parameter (σ)
  - Formula: W(t) = μt + σB(t)

- **`geometric_brownian(n, T, k, a, l)`**: Geometric Brownian motion
  - `a`: Drift coefficient
  - `l`: Diffusion coefficient
  - Formula: W(t) = exp(at + λB(t))

#### Example

```python
from Brownian import standard_brownian
import matplotlib.pyplot as plt

# Generate 1000 paths of standard Brownian motion
simulations, rand_num, time = standard_brownian(n=500, T=1, k=1000)

# Plot the paths
for simulation in simulations:
    plt.plot(time, simulation, linewidth=1)
plt.show()
```

### Numerical Methods

**File**: `Numerical.py`

Implements numerical methods for solving stochastic differential equations.

#### Functions

- **`homogeneous_equation(x_0, u, s, n, m, bt, time)`**: Exact solution for geometric Brownian motion
  - Solves: dX(t) = μX(t)dt + σX(t)dB(t)
  - Solution: X(t) = X₀ exp((μ - σ²/2)t + σB(t))
  - `x_0`: Initial value
  - `u`: Drift parameter (μ)
  - `s`: Volatility parameter (σ)
  - `m`: Number of paths
  - Returns: Array of solution paths

- **`euler(x_0, u, s, n, b1)`**: Euler-Maruyama method
  - First-order approximation: X(t+dt) = X(t) + μX(t)dt + σX(t)√dt·Z
  - Convergence: O(√dt)

- **`milstein(x_0, u, s, n, b1)`**: Milstein method
  - Second-order approximation with correction term
  - Convergence: O(dt)
  - More accurate than Euler for the same step size

#### Example

```python
from Brownian import standard_brownian
from Numerical import homogeneous_equation, euler, milstein
import matplotlib.pyplot as plt

x_0, u, s, n = 1, 1.5, 2.5, 500
bt, [b1], time = standard_brownian(n, 1, 1)

# Compare exact solution with numerical methods
x_exact = homogeneous_equation(x_0, u, s, n, 1, bt, time)
x_euler = euler(x_0, u, s, n, b1)
x_milstein = milstein(x_0, u, s, n, b1)

plt.plot(time, x_exact[0], label='Exact')
plt.plot(time, x_euler, label='Euler')
plt.plot(time, x_milstein, label='Milstein')
plt.legend()
plt.show()
```

### Parameter Estimation

**File**: `Estimation.py`

Implements parameter estimation for stochastic processes using maximum likelihood estimation.

#### Functions

- **`instant_returns(n, x_t)`**: Calculates instantaneous returns
  - r(t) = (X(t) - X(t-1)) / X(t-1)

#### Main Features

- Estimates drift (μ) and volatility (σ) parameters from simulated paths
- Constructs confidence intervals using normal approximation
- Validates estimation accuracy through multiple simulations

#### Example

```python
from Estimation import instant_returns

# Calculate returns from price series
returns = instant_returns(n, price_series)

# Estimate parameters
dt = 1/n
u_hat = np.mean(returns) / dt
s_hat = np.sqrt(np.var(returns, ddof=0) / dt)
```

### Option Valuation

**File**: `OptionValuation.py`

Implements four different methods for pricing European call options.

#### Functions

- **`black_scholes(x_0, r, s, k, T)`**: Analytical Black-Scholes formula
  - Exact solution for European options
  - Formula: C = S₀N(d₁) - Ke^(-rT)N(d₂)
  - Fastest method

- **`montecarlo(x_0, r, s, n, T, m, w, k)`**: Monte Carlo simulation
  - Simulates stock price paths
  - Averages discounted payoffs
  - Parameters:
    - `m`: Number of paths per simulation
    - `w`: Number of simulations
  - Good for complex options

- **`binomial_tree(x_0, r, s, k, T, n)`**: Binomial tree method
  - Discrete-time lattice approach
  - Backward induction from maturity
  - Good for American options (extensible)

- **`finite_differences(s, r, k, T, m)`**: Finite difference method
  - Solves Black-Scholes PDE numerically
  - Uses theta-method for time stepping
  - Stability condition: dt ≤ 0.9/(σ²m²)

#### Comparison

| Method | Speed | Accuracy | Complexity |
|--------|-------|----------|------------|
| Black-Scholes | ⚡⚡⚡ | Exact | Low |
| Monte Carlo | 🐌 | Statistical | Medium |
| Binomial Tree | ⚡ | Good | Medium |
| Finite Differences | ⚡⚡ | Very Good | High |

#### Example

```python
from OptionValuation import black_scholes, montecarlo, binomial_tree, finite_differences

x_0 = 100    # Current stock price
r = 0.05     # Risk-free rate
s = 0.3      # Volatility
k = 100      # Strike price
T = 1        # Time to maturity

# Compare all methods
price_bs = black_scholes(x_0, r, s, k, T)
price_mc, _ = montecarlo(x_0, r, s, 500, T, 1000, 100, k)
price_bt = binomial_tree(x_0, r, s, k, T, 500)
price_fd = finite_differences(s, r, k, T, 500)

print(f"Black-Scholes: {price_bs:.4f}")
print(f"Monte Carlo: {price_mc:.4f}")
print(f"Binomial Tree: {price_bt:.4f}")
print(f"Finite Diff: {price_fd:.4f}")
```

### Utility Functions

**File**: `Functions.py`

Helper functions for array operations.

- **`vector_times_scalars(vector, scalars)`**: Multiplies a vector by an array of scalars
  - Used in Brownian bridge construction

### Statistical Testing

**File**: `StatTest.r`

R script for testing normality of stock returns.

#### Features

- Downloads historical stock data using `BatchGetSymbols`
- Performs normality tests:
  - **Lilliefors test**: Modified Kolmogorov-Smirnov test
  - **Jarque-Bera test**: Tests skewness and kurtosis
  - **Shapiro-Wilk test**: General normality test
- Generates histograms and Q-Q plots
- Exports selected data to CSV

#### Usage

```r
source("StatTest.r")
```

The script will:
1. Download data for multiple NASDAQ tickers (2000-2019)
2. Test each year for each ticker
3. Display histograms for data passing normality tests
4. Export AABA 2019 data for further analysis

## Usage Examples

### Example 1: Simulating Stock Prices

```python
from Brownian import standard_brownian
from Numerical import homogeneous_equation
import matplotlib.pyplot as plt

# Parameters for stock price
S0 = 100        # Initial price
mu = 0.1        # Expected return (10%)
sigma = 0.2     # Volatility (20%)
T = 1           # 1 year
n = 252         # Trading days
k = 100         # Number of simulations

# Simulate Brownian motion
bt, _, time = standard_brownian(n, T, k)

# Simulate stock prices
prices = homogeneous_equation(S0, mu, sigma, n, k, bt, time)

# Plot
for price in prices:
    plt.plot(time, price, alpha=0.3, linewidth=0.5)
plt.xlabel('Time (years)')
plt.ylabel('Stock Price ($)')
plt.title('Stock Price Simulation (GBM)')
plt.show()
```

### Example 2: Parameter Estimation with Confidence Intervals

```python
from Brownian import standard_brownian
from Numerical import homogeneous_equation
from Estimation import instant_returns
import numpy as np
from scipy.stats import norm

# True parameters
x_0, u, s = 1, 1.5, 2.5
n, k = 500, 500
alpha = 0.05

# Run multiple estimations
u_estimates, s_estimates = [], []

for i in range(k):
    [bt], [b1], time = standard_brownian(n, 1, 1)
    x_t = homogeneous_equation(x_0, u, s, n, 1, bt, time)

    returns = instant_returns(n, x_t[0])
    dt = 1/n

    u_hat = np.mean(returns) / dt
    s_hat = np.sqrt(np.var(returns, ddof=0) / dt)

    u_estimates.append(u_hat)
    s_estimates.append(s_hat)

# Calculate confidence intervals
u_mean = np.mean(u_estimates)
s_mean = np.mean(s_estimates)

z = norm.ppf(1 - alpha/2)
u_ci = (u_mean - z*np.std(u_estimates)/np.sqrt(k),
        u_mean + z*np.std(u_estimates)/np.sqrt(k))
s_ci = (s_mean - z*np.std(s_estimates)/np.sqrt(k),
        s_mean + z*np.std(s_estimates)/np.sqrt(k))

print(f"True μ = {u}, Estimated μ = {u_mean:.4f}, 95% CI: {u_ci}")
print(f"True σ = {s}, Estimated σ = {s_mean:.4f}, 95% CI: {s_ci}")
```

### Example 3: Option Pricing Sensitivity Analysis

```python
from OptionValuation import black_scholes
import numpy as np
import matplotlib.pyplot as plt

S0 = 100
r = 0.05
sigma = 0.3
T = 1

# Strike price sensitivity
strikes = np.linspace(80, 120, 50)
prices = [black_scholes(S0, r, sigma, K, T) for K in strikes]

plt.figure(figsize=(10, 6))
plt.plot(strikes, prices)
plt.xlabel('Strike Price ($)')
plt.ylabel('Option Price ($)')
plt.title('European Call Option Price vs Strike')
plt.grid(True)
plt.show()

# Volatility sensitivity (Vega)
volatilities = np.linspace(0.1, 0.6, 50)
prices = [black_scholes(S0, r, vol, S0, T) for vol in volatilities]

plt.figure(figsize=(10, 6))
plt.plot(volatilities, prices)
plt.xlabel('Volatility (σ)')
plt.ylabel('Option Price ($)')
plt.title('European Call Option Price vs Volatility')
plt.grid(True)
plt.show()
```

## Main Analysis Pipeline

**File**: `Final.py`

The main analysis script that demonstrates all functionality:

### Part 1: Stochastic Process Simulation

#### 1.1: Geometric Brownian Motion with Predictions

- Simulates stock prices using GBM
- Creates three future scenarios (bearish, neutral, bullish)
- Constructs prediction bands using Monte Carlo
- Validates prediction accuracy

#### 1.2: CEV Process (Constant Elasticity of Variance)

- Simulates CEV process: dX = a(μ - X)dt + σX^r dB(t)
- Estimates parameters from simulated data
- Tests parameter sensitivity

**Functions:**
- `punto1_1_1`: Single path simulation
- `punto1_1_2`: Three scenario projections
- `punto1_1_3`: Prediction band construction and validation
- `punto1_1_4`: Multiple volatility testing
- `punto1_2_1`: CEV simulation
- `punto1_2_3`: CEV parameter estimation
- `punto1_2_4`: Parameter sensitivity analysis

### Part 2: Real Data Analysis and Option Pricing

Uses real stock data (AABA 2019) to:

- Estimate risk-free rate and volatility from returns
- Price options using all four methods
- Perform sensitivity analysis on:
  - Time to maturity (T)
  - Risk-free rate (r)
  - Volatility (σ)
  - Strike price (K)
  - Monte Carlo parameters (m, w, n)
  - Grid size for numerical methods

**Functions:**
- `punto2_1_2`: Parameter estimation from real data
- `punto2_1_3`: Four-method option pricing
- `punto2_2_1` to `punto2_2_7`: Sensitivity analyses

### Running the Complete Analysis

```bash
python Final.py
```

This will:
1. Run all simulations with visualizations
2. Estimate parameters with confidence intervals
3. Price options using all methods
4. Generate sensitivity analysis results

## Mathematical Background

### Stochastic Differential Equations

#### Geometric Brownian Motion (GBM)

The fundamental model for stock prices:

```
dS(t) = μS(t)dt + σS(t)dB(t)
```

Exact solution:
```
S(t) = S₀ exp((μ - σ²/2)t + σB(t))
```

Where:
- μ: drift (expected return)
- σ: volatility (standard deviation of returns)
- B(t): Standard Brownian motion

#### Black-Scholes Partial Differential Equation

```
∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
```

With boundary condition: V(S,T) = max(S-K, 0) for a call option

### Numerical Methods

#### Euler-Maruyama Method

```
X_{n+1} = X_n + μX_n Δt + σX_n√Δt Z_n
```

Where Z_n ~ N(0,1)

#### Milstein Method

Includes second-order correction:
```
X_{n+1} = X_n + μX_n Δt + σX_n√Δt Z_n + ½σ²X_n((√Δt Z_n)² - Δt)
```

### Option Pricing

#### Black-Scholes Formula

For a European call option:

```
C = S₀N(d₁) - Ke^(-rT)N(d₂)

d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

Where N(·) is the cumulative standard normal distribution.

#### Monte Carlo Pricing

1. Simulate M paths of S(t)
2. Calculate payoff: max(S(T) - K, 0)
3. Average and discount: C = e^(-rT) · E[max(S(T) - K, 0)]

## Requirements

### Python Packages

Create a `requirements.txt` file:

```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0
```

Install with:
```bash
pip install -r requirements.txt
```

### R Packages

```r
# Financial data
install.packages("BatchGetSymbols")

# Statistical tests
install.packages("nortest")
install.packages("tseries")

# Visualization
install.packages("ggpubr")
```

## Data

### AABA_2019.csv

Stock price data for Altaba Inc. (formerly Yahoo!) for the year 2019, including:
- Date
- Open, High, Low, Close prices
- Adjusted prices
- Volume
- Returns (ret.adjusted.prices)

This data is used for:
- Parameter estimation (drift and volatility)
- Real-world option pricing
- Validation of pricing models

The R script `StatTest.r` can be used to download and test data for other tickers.

## Performance Notes

### Computational Complexity

- **Black-Scholes**: O(1) - Instant
- **Monte Carlo**: O(m × w × n) - Slow for high accuracy
- **Binomial Tree**: O(n²) - Moderate
- **Finite Differences**: O(m × n) - Moderate to fast

### Recommended Parameters

For quick testing:
```python
n = 100      # Time steps
m = 100      # Paths/Grid points
w = 10       # MC simulations
```

For production:
```python
n = 500      # Time steps
m = 1000     # Paths/Grid points
w = 100      # MC simulations
```
