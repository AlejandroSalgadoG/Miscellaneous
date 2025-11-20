# Generalized Gamma Distribution - Beamer Presentation

**Author:** Alejandro Salgado Gómez
**Document Type:** LaTeX Beamer Presentation (Spanish)

## Overview

This presentation provides a comprehensive mathematical treatment of the **Generalized Gamma Distribution**, including its theoretical foundations, properties, relationships with other distributions, and practical applications.

## Contents

### 1. Probability Density Function (PDF)
The presentation defines a random variable following a Generalized Gamma distribution with parameters α, β, and ρ:

```
f(x) = (x^(α-1) * e^(-(x/β)^ρ)) / (Γ(α/ρ) * β^α/ρ)
```

Where:
- **x, α, β, ρ > 0**
- **α and ρ**: Shape parameters
- **β**: Scale parameter

### 2. Cumulative Distribution Function (CDF)
Includes a detailed step-by-step mathematical derivation showing:
- Integration by parts
- Variable substitution: `y = (t/β)^ρ`
- Connection to the incomplete gamma function
- Final result: `F(x) = Γ(α/ρ, (x/β)^ρ) / Γ(α/ρ)`

### 3. Moment Generating Function
The moment generating function is presented as:

```
M(θ) = Σ(r=0 to ∞) [(θβ)^r / r!] * [Γ((α+r)/ρ) / Γ(α/ρ)]
```

### 4. Moments
Detailed derivations of:
- **General n-th moment**: `E[x^n] = [Γ((α+n)/ρ) / Γ(α/ρ)] * β^n`
- **Expected value**: `E[x] = [Γ((α+1)/ρ) / Γ(α/ρ)] * β`
- **Variance**: `V[x] = β² * [Γ((α+2)/ρ)/Γ(α/ρ) - Γ((α+1)/ρ)²/Γ(α/ρ)²]`
- **Skewness** (α₃): Detailed formula in terms of Gamma functions
- **Kurtosis** (α₄): Detailed formula in terms of Gamma functions

### 5. Relationships with Other Distributions

The Generalized Gamma distribution generalizes several well-known distributions:

| Condition | Resulting Distribution |
|-----------|------------------------|
| ρ = 1 | **Gamma distribution** with parameters α and β |
| ρ = 1, α = n (n ∈ ℕ) | **Erlang distribution** with parameters n and 1/β |
| ρ = 1, α = v/2, β = 2 (v ∈ ℤ⁺) | **Chi-square distribution** with v degrees of freedom |
| ρ = 1, α = 1 | **Exponential distribution** with parameter β |
| ρ = α | **Weibull distribution** with parameters α and β |

**Special property**: If `X_i ~ GenGamma(α_i, β_i, ρ_i)` for i=1,...,n, then:
```
Y = Σ(X_i/β_i)^ρ_i ~ GenGamma(1, Σ(α_i/ρ_i), 1)
```

### 6. Random Variable Generation

The presentation covers three methods for generating random variables:

#### R Package: flexsurv
```R
rgengamma.orig(n, b, a, k)
```
- n: number of samples
- b: ρ (shape parameter)
- a: β (scale parameter)
- k: α/ρ

#### R Package: rmutil
```R
rggamma(n, σ, μ, v)
```
- n: number of samples
- σ: α/ρ
- μ: βα/ρ
- v: ρ

#### Mathematica
```mathematica
RandomVariate[GammaDistribution[θ, β, ρ, μ], n]
```
- θ: α/ρ
- μ: location parameter (x > μ)
- β: scale parameter
- ρ: shape parameter

### 7. Applications

The Generalized Gamma distribution has applications in:

1. **Engineering and Survival Analysis** - For modeling extreme values and reliability
2. **Financial Risk Assessment** - Portfolio risk evaluation and investment analysis
3. **Economic Theory** - Specifically in risk theory
4. **Actuarial Science** - Insurance and risk management

## Key References

- **Stacy, E. W. (1962)**: "A Generalization of the Gamma Distribution" - The Annals of Mathematical Statistics
- **Tadikamalla, P.R. (1978)**: "Random sampling from the generalized gamma distribution"
- **Du, Lingling & Chen, Shouquan (2016)**: "Asymptotic properties for distributions and densities of extremes from generalized gamma distribution"
- Applications in risk theory, actuarial mathematics, and financial modeling

## File Structure

- `Presentation.tex` - Main presentation file
- `Pdf.tex` - Probability density function content
- `Cdf.tex` - Cumulative distribution function derivation
- `MGenerating.tex` - Moment generating function
- `Moments.tex` - Moments calculations
- `Relations.tex` - Relationships with other distributions
- `Generation.tex` - Random variable generation methods
- `Application.tex` - Real-world applications
- `Presentation.bib` - Bibliography references
- `Makefile` - Build automation

## Building the Presentation

Use the included Makefile to compile the presentation:

```bash
make
```

This will generate the PDF output of the Beamer presentation.

## Mathematical Notation

- **Γ(x)**: Gamma function
- **Γ(a, x)**: Incomplete gamma function
- **E[x]**: Expected value
- **V[x]**: Variance
- **α₃**: Skewness coefficient
- **α₄**: Kurtosis coefficient

## Language

The presentation is written in **Spanish** (`\usepackage[spanish]{babel}`).

---

*This presentation provides a rigorous mathematical treatment suitable for graduate-level statistics, actuarial science, or applied mathematics courses.*
