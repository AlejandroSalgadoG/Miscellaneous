# MultStatistic

A Python implementation for multivariate statistical analysis, contains various techniques for discriminant analysis, hypothesis testing, outlier detection, principal component analysis, and regression.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Modules](#modules)
  - [1. Discriminant Analysis](#1-discriminant-analysis)
  - [2. Hypothesis Testing](#2-hypothesis-testing)
  - [3. Outlier Detection](#3-outlier-detection)
  - [4. Principal Component Analysis (PCA)](#4-principal-component-analysis-pca)
  - [5. Regression Analysis](#5-regression-analysis)
- [Dependencies](#dependencies)
- [Mathematical Background](#mathematical-background)
- [Contributing](#contributing)
- [License](#license)

## Overview

MultStatistic is a collection of Python implementations for common multivariate statistical methods. It is good for

- Classifying observations into groups
- Testing hypotheses about multivariate means
- Detecting outliers in multivariate data
- Reducing dimensionality through PCA
- Building and testing multiple regression models

## Installation

```bash
pip install -r requirements.txt
```

## Dependencies

- `numpy` - Numerical computing
- `pandas` - Data manipulation and analysis
- `scipy` - Scientific computing and statistical functions
- `matplotlib` - Data visualization
- `scikit-learn` - Machine learning algorithms

Create a `requirements.txt` file with:
```
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
```

## Modules

### 1. Discriminant Analysis

Location: `DiscriminantAna/`

#### Linear Discriminant Analysis (LDA)

**File:** `Discriminant.py`

Performs Fisher's Linear Discriminant Analysis to find linear combinations of features that best separate multiple classes.

**Features:**
- Computes between-group scatter matrix (H)
- Computes within-group scatter matrix (E)
- Extracts discriminant functions via eigenvalue decomposition
- Projects data onto discriminant axes

**Mathematical Foundation:**
The discriminant analysis solves the eigenvalue problem:
```
E^(-1)H·A = λA
```
where λ are eigenvalues and A are eigenvectors (discriminant coefficients).

**Usage Example:**
```python
import pandas as pd
import numpy as np

# Load grouped data
data = pd.read_csv("data.csv")
groups = [group.reset_index(drop=True).drop(columns=["G"])
          for _, group in data.groupby("G")]

# Parameters
k = 3  # number of groups
n = 30 # observations per group
p = 6  # number of variables

# Perform LDA
# ... (see Discriminant.py for full implementation)
```

#### Classification

**File:** `Classification.py`

Implements both linear and quadratic discriminant analysis for classification.

**Methods:**
1. **Linear Classification Rule:**
   - Assumes equal covariance matrices across groups
   - Uses pooled covariance matrix
   - Classification criterion: \( \log(1/k) - a/2 + b^T y \)

2. **Quadratic Classification Rule:**
   - Allows different covariance matrices per group
   - Uses Mahalanobis distance
   - Classification criterion: \( \log(1/k) - \log|S|/2 + d_M^2(y, \bar{y})/2 \)

**Usage Example:**
```python
from scipy.spatial.distance import mahalanobis

# Linear classification
def lin_classif(y_bar, Sp_1, y):
    b = y_bar.T.dot(Sp_1)
    a = b.dot(y_bar)
    return np.log(1/k) - a/2 + b.dot(y)

# Quadratic classification
def quad_classif(y_bar, S, y):
    S_1 = np.linalg.inv(S.values)
    d_S = np.linalg.det(S.values)
    return np.log(1/k) - np.log(d_S)/2 + mahalanobis(y, y_bar, S_1)**2/2
```

---

### 2. Hypothesis Testing

Location: `HipTest/`

#### Comparing Two Multivariate Means

**File:** `CompareMean.py`

Tests whether two multivariate populations have equal mean vectors using Hotelling's T² test.

**Statistical Test:**
- Null Hypothesis: \( H_0: \mu_1 = \mu_2 \)
- Test Statistic: \( T^2 = \frac{n_1 n_2}{n_1 + n_2} d_M^2(\bar{y}_1, \bar{y}_2) \)
- F-statistic: \( F = \frac{n_1 + n_2 - p - 1}{(n_1 + n_2 - 2)p} T^2 \)
- Distribution: \( F \sim F_{p, n_1+n_2-p-1} \)

**Output:**
- Discriminant coefficient vector
- F-test statistic and critical value
- Visualization of F-distribution with test statistic

**Usage:**
```python
# Define two samples X1 and X2
# Compute pooled covariance
sp = (n1-1)/(n1+n2-2) * s1 + (n2-1)/(n1+n2-2) * s2

# Calculate T² statistic
t_2 = (n1*n2)/(n1+n2) * mahalanobis(y1_bar, y2_bar, sp_1)**2

# Convert to F-statistic
F_test = (n1+n2-p-1)/((n1+n2-2)*p) * t_2
```

#### Test with Known Covariance (Σ)

**File:** `TestUZknown.py`

Tests hypotheses about a multivariate mean when the covariance matrix is known.

**Statistical Test:**
- Test Statistic: \( Z^2 = n(\bar{y} - \mu_0)^T \Sigma^{-1} (\bar{y} - \mu_0) \)
- Distribution: \( Z^2 \sim \chi^2_p \)
- Used when population covariance is known from prior studies

**Application:** Table 3.1 example with height and weight data

#### Test with Unknown Covariance (Σ)

**File:** `TestUZunknown.py`

Tests hypotheses about a multivariate mean when the covariance matrix is unknown.

**Statistical Test:**
- Test Statistic: \( T^2 = n(\bar{y} - \mu_0)^T S^{-1} (\bar{y} - \mu_0) \)
- F-statistic: \( F = \frac{n-p}{(n-1)p} T^2 \)
- Distribution: \( F \sim F_{p, n-p} \)
- Sample covariance S is used instead of Σ

**Application:** Table 3.3 example with three-variable data

---

### 3. Outlier Detection

Location: `OutlierDetect/`

Implements robust methods for detecting outliers in multivariate data.

#### Mahalanobis Distance Plot

**File:** `PlotMahalDist.py`

Visualizes Mahalanobis distances for each observation to identify potential outliers.

**Method:**
- Computes Mahalanobis distance: \( d_M(x) = \sqrt{(x - \bar{x})^T S^{-1} (x - \bar{x})} \)
- Threshold: \( \sqrt{\chi^2_{p,\alpha}} \)
- Points exceeding threshold are potential outliers

**Usage:**
```python
# Compute Mahalanobis distances
dist = [mahalanobis(x, u_bar, s_1) for _, x in data.iterrows()]

# Set threshold (e.g., 95% confidence)
threshold = np.sqrt(chi2.ppf(0.95, p))

# Plot
plt.scatter(np.arange(n), dist)
plt.hlines(threshold, 0, n)
```

#### Minimum Covariance Determinant (MCD)

**File:** `MinimumCovarianceDeterminant.py`

Robust estimation of location and scatter that minimizes the determinant of the covariance matrix.

**Algorithm:**
1. Generate random subsets of size h = ⌊(n+p+1)/2⌋
2. For each subset:
   - Compute mean and covariance
   - Perform C-steps (concentration steps)
   - Compute objective function: det(S)
3. Select subset with minimum determinant

**Advantages:**
- High breakdown point (up to 50% contamination)
- Resistant to outliers
- Provides robust estimates of center and spread

**Parameters:**
- `m = 3000`: Number of random subsets
- `h = floor((n+p+1)/2)`: Size of each subset
- `c_steps = 5`: Number of concentration steps

#### Minimum Volume Ellipsoid (MVE)

**File:** `MinimumVolumeEllipsoid.py`

Finds the ellipsoid of minimum volume that covers at least h observations.

**Algorithm:**
1. Generate random subsets of size p+1
2. For each subset:
   - Compute mean, covariance, and determinant
   - Sort Mahalanobis distances
   - Compute objective function: \( (d_h/c)^p \sqrt{\det(S)} \)
3. Select subset minimizing the objective function

**Objective Function:**
- Minimizes volume while covering h observations
- \( c = \sqrt{\chi^2_{p,h/n}} \) is a correction factor

#### Ellipse Visualization

**File:** `Ellipse.py`

Creates tolerance ellipses for bivariate data visualization.

**Method:**
- Computes eigenvalues and eigenvectors of covariance matrix
- Ellipse dimensions: \( \sqrt{\lambda_i \cdot \chi^2_{p,\alpha}} \)
- Rotation angle from principal eigenvector
- Useful for visualizing data spread and correlation

#### Auxiliary Functions

**File:** `Auxiliar.py`

Helper functions for subset generation:
- `check_duplicates()`: Ensures subset uniqueness
- `random_subsets()`: Generates random subsets without replacement

---

### 4. Principal Component Analysis (PCA)

Location: `Pca/`

**File:** `Pca.py`

Performs dimensionality reduction by finding orthogonal linear combinations of variables that capture maximum variance.

**Features:**
- Eigenvalue decomposition of covariance matrix
- Computation of principal components
- Variance proportion explained by each component
- Data projection onto principal component space

**Mathematical Foundation:**
Given covariance matrix Σ:
1. Solve: \( \Sigma A = \lambda A \)
2. Sort eigenvalues: \( \lambda_1 \geq \lambda_2 \geq ... \geq \lambda_p \)
3. Proportion of variance: \( \lambda_i / \sum \lambda_j \)
4. Transform data: \( Z = X A \)

**Usage Example:**
```python
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv("Data.txt")
data = data[["WDIM", "CIRCUM", "FBEYE", "EYEHD", "EARHD", "JAW"]]

# Compute covariance matrix
cov = data.cov()

# Eigenvalue decomposition
l, A = np.linalg.eig(cov)
idx = np.argsort(l)[::-1]
l, A = l[idx], A[:, idx]

# Proportion of variance explained
prop_var = l / np.sum(l)

# Transform to principal components
Z = data.dot(A)
```

**Interpretation:**
- First PC captures direction of maximum variance
- Subsequent PCs are orthogonal and capture remaining variance
- Standard deviations: \( \sqrt{\lambda_i} \)
- Loadings matrix A shows variable contributions

---

### 5. Regression Analysis

Location: `Regression/`

#### Multiple Linear Regression

**File:** `Regression.py`

Fits a multiple linear regression model and performs diagnostic checks.

**Model:**
\[ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p + \epsilon \]

**Features:**
- Ordinary Least Squares (OLS) estimation
- 3D visualization of regression plane (for 2 predictors)
- Residual analysis
- Normality test (Jarque-Bera)

**Estimation:**
\[ \hat{\beta} = (X^T X)^{-1} X^T y \]

**Usage Example:**
```python
import pandas as pd
from numpy.linalg import inv

# Load data
data = pd.read_csv("marketing.csv")
data["bias"] = 1

y = data["sales"]
x = data[["bias", "youtube", "newspaper"]]

# Estimate coefficients
b = inv(x.T.dot(x)).dot(x.T).dot(y)

# Predictions
y_hat = x.dot(b)

# Residuals
error = y - y_hat
```

#### Hypothesis Testing for Regression

**File:** `Test.py`

Performs statistical tests for regression coefficients and overall model fit.

**Tests Performed:**

1. **Overall F-test:**
   - \( H_0 \): All coefficients are zero (except intercept)
   - \( F = \frac{MS_{reg}}{MS_{error}} \)
   - Distribution: \( F \sim F_{p, n-p-1} \)

2. **Individual t-tests:**
   - \( H_0: \beta_j = 0 \)
   - \( t_j = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)} \)
   - Distribution: \( t \sim t_{n-p-1} \)

**Sums of Squares:**
- SSR (Regression): \( \sum (\hat{y}_i - \bar{y})^2 \)
- SSE (Error): \( \sum (y_i - \hat{y}_i)^2 \)
- SST (Total): \( \sum (y_i - \bar{y})^2 \)

**Standard Errors:**
\[ SE(\hat{\beta}) = \sqrt{MS_{error} \cdot \text{diag}((X^T X)^{-1})} \]

**Usage Example:**
```python
from scipy.stats import t, f

# Compute test statistics
f_stad = ms_reg / ms_error
t_stad = b / se_b

# Critical values
f_crit = f.ppf(1 - alpha, dof_reg, dof_fit)
t_crit = t.ppf(1 - alpha/2, dof_fit)
```

---

## Mathematical Background

### Multivariate Normal Distribution

Most methods assume data follows a multivariate normal distribution:
\[ X \sim N_p(\mu, \Sigma) \]

where:
- \( \mu \) is the p-dimensional mean vector
- \( \Sigma \) is the p×p covariance matrix

### Mahalanobis Distance

A measure of distance accounting for correlation structure:
\[ d_M(x, \mu) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)} \]

Under normality: \( d_M^2 \sim \chi^2_p \)

### Hotelling's T² Distribution

Generalization of Student's t to multivariate case:
\[ T^2 = n(\bar{X} - \mu_0)^T S^{-1} (\bar{X} - \mu_0) \]

Related to F-distribution:
\[ F = \frac{n-p}{(n-1)p} T^2 \sim F_{p, n-p} \]

### Eigenvalue Decomposition

Central to many multivariate methods:
\[ \Sigma = A \Lambda A^T \]

where:
- A contains eigenvectors (orthonormal)
- Λ is diagonal matrix of eigenvalues

---

## Project Structure

```
MultStatistic/
├── DiscriminantAna/
│   ├── Discriminant.py       # Linear Discriminant Analysis
│   ├── Classification.py     # LDA and QDA classification
│   └── data.csv             # Sample dataset
├── HipTest/
│   ├── CompareMean.py       # Two-sample mean comparison
│   ├── TestUZknown.py       # Test with known covariance
│   └── TestUZunknown.py     # Test with unknown covariance
├── OutlierDetect/
│   ├── Auxiliar.py          # Helper functions
│   ├── Ellipse.py           # Tolerance ellipse plotting
│   ├── MinimumCovarianceDeterminant.py  # MCD estimator
│   ├── MinimumVolumeEllipsoid.py        # MVE estimator
│   └── PlotMahalDist.py     # Mahalanobis distance plot
├── Pca/
│   └── Pca.py               # Principal Component Analysis
├── Regression/
│   ├── Regression.py        # Multiple linear regression
│   └── Test.py              # Hypothesis tests for regression
└── README.md
```

## Usage Tips

1. **Data Preparation:**
   - Ensure data is in proper format (pandas DataFrame or numpy array)
   - Check for missing values
   - Consider standardization for methods sensitive to scale

2. **Outlier Detection:**
   - Start with Mahalanobis distance plot for quick overview
   - Use MCD or MVE for robust estimation when outliers suspected
   - Compare classical vs. robust estimates

3. **Dimension Reduction:**
   - Use PCA when variables are highly correlated
   - LDA when primary goal is classification/discrimination
   - Check proportion of variance explained to select components

4. **Hypothesis Testing:**
   - Verify multivariate normality assumption
   - Check sample size requirements (n >> p)
   - Consider power analysis for study design

5. **Regression:**
   - Examine residual plots for assumptions
   - Test for multicollinearity (VIF)
   - Perform cross-validation for predictive accuracy
