# Flowshop Scheduling Problem - Heuristic Solutions

implementation of various heuristic and metaheuristic algorithms to solve the **Permutation Flowshop Scheduling Problem (PFSP)**. This project was developed as part of research on optimization algorithms and includes implementations of Genetic Algorithms, Iterated Local Search (ILS), Variable Neighborhood Descent (VND), and GRASP construction methods.

## Table of Contents

- [Problem Description](#problem-description)
- [Algorithms Implemented](#algorithms-implemented)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Problem Instances](#problem-instances)
- [Results](#results)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [References](#references)

## Problem Description

The **Permutation Flowshop Scheduling Problem (PFSP)** is a classic NP-Hard optimization problem in operations research. In this problem:

- **n jobs** must be processed on **m machines**
- Each job requires processing on every machine in the same order
- Each machine can process only one job at a time
- Jobs follow the same permutation order on all machines
- Processing times are known and deterministic

**Objective**: Find the permutation of jobs that minimizes the total flowtime (sum of completion times of all jobs).

### Mathematical Formulation

Given:
- `n` jobs and `m` machines
- Processing time `p[i][j]` for job `i` on machine `j`

Find a permutation π of jobs that minimizes:

```
Total Flowtime = Σ(C[π[i]][m]) for all jobs i
```

where `C[i][j]` is the completion time of job `i` on machine `j`.

## Algorithms Implemented

This project implements four different approaches to solve the PFSP:

### 1. Genetic Algorithm (`/Genetic`)

A population-based evolutionary algorithm that mimics natural selection.

**Features:**
- Random initial population generation
- Crossover operator for breeding new solutions
- Mutation operator for exploration
- Local search integration for intensification
- Elitist selection strategy

**Key Components:**
- `Genetic.py`: Core genetic algorithm implementation
- `Main.py`: Entry point with parameter configuration
- `Calc_cost.py`: Objective function evaluation
- `Reader.py`: Problem instance reader

### 2. Iterated Local Search - ILS (`/Local_search`)

An iterative improvement algorithm that alternates between perturbation and local search phases.

**Features:**
- Best insertion construction for initial solution
- Local search for intensification
- Perturbation mechanism to escape local optima
- Iterative refinement

**Key Components:**
- `Ils.py`: ILS implementation with swap-based perturbation
- `Local_search.py`: Local optimization procedures
- `Best_insertion.py`: Greedy constructive heuristic
- `Main.py`: Execution entry point

### 3. Variable Neighborhood Descent - VND (`/Neighborhood`)

A systematic exploration of multiple neighborhood structures.

**Features:**
- Multiple neighborhood structures
- Deterministic neighborhood exploration
- Best insertion for initial solution
- Two different local search operators

**Key Components:**
- `Vnd.py`: VND implementation with neighborhood switching
- `Local_search.py`: Multiple neighborhood operators
- `Best_insertion.py`: Constructive heuristic

### 4. GRASP Construction (`/Random`)

A greedy randomized adaptive search procedure for solution construction.

**Features:**
- Randomized greedy construction
- Restricted Candidate List (RCL)
- Adaptive candidate selection
- Alpha parameter for diversification control

**Key Components:**
- `Grasp_construction.py`: GRASP-based constructive heuristic
- `Noice.py`: Additional randomization utilities
- `Reader.py`: Problem instance reader

## Project Structure

```
Heuristics/
├── Article/                  # Research paper (LaTeX)
│   ├── Trabajo4.tex         # Main document
│   ├── Trabajo4.bib         # Bibliography
│   └── Makefile             # LaTeX compilation
│
├── Genetic/                 # Genetic Algorithm implementation
│   ├── Main.py              # Entry point
│   ├── Genetic.py           # GA core logic
│   ├── Calc_cost.py         # Cost calculation
│   ├── Reader.py            # Instance reader
│   ├── execute.sh           # Single execution script
│   └── execute_all.sh       # Batch execution script
│
├── Local_search/            # Iterated Local Search (ILS)
│   ├── Main.py              # Entry point
│   ├── Ils.py               # ILS implementation
│   ├── Local_search.py      # Local search operators
│   ├── Best_insertion.py    # Constructive heuristic
│   ├── Calc_cost.py         # Cost calculation
│   └── Reader.py            # Instance reader
│
├── Neighborhood/            # Variable Neighborhood Descent (VND)
│   ├── Main.py              # Entry point
│   ├── Vnd.py               # VND implementation
│   ├── Local_search.py      # Multiple neighborhood operators
│   ├── Best_insertion.py    # Constructive heuristic
│   ├── Calc_cost.py         # Cost calculation
│   └── Reader.py            # Instance reader
│
├── Random/                  # GRASP construction
│   ├── Grasp_construction.py # GRASP implementation
│   ├── Noice.py             # Noise/randomization utilities
│   └── Reader.py            # Instance reader
│
├── ProblemGen/              # Problem instance generator
│   ├── ProblemGen.c         # C implementation (Taillard's method)
│   ├── LowerBound.py        # Lower bound calculator
│   └── Makefile             # Compilation script
│
└── Instances/               # Test instances
    ├── 20_5.txt             # 20 jobs, 5 machines
    ├── 20_10.txt            # 20 jobs, 10 machines
    ├── 20_20.txt            # 20 jobs, 20 machines
    ├── 50_5.txt             # 50 jobs, 5 machines
    ├── 50_10.txt            # 50 jobs, 10 machines
    ├── 50_20.txt            # 50 jobs, 20 machines
    ├── 100_5.txt            # 100 jobs, 5 machines
    ├── 100_10.txt           # 100 jobs, 10 machines
    ├── 100_20.txt           # 100 jobs, 20 machines
    └── Best_solutions.txt   # Known best solutions
```

## Installation

### Prerequisites

- **Python 3.6+** (for heuristic implementations)
- **NumPy** (numerical computations)
- **GCC** (for problem instance generator)
- **LaTeX** (optional, for article compilation)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Heuristics
```

2. Install Python dependencies:
```bash
pip install numpy
```

3. (Optional) Compile the problem generator:
```bash
cd ProblemGen
make
cd ..
```

## Usage

### Genetic Algorithm

```bash
cd Genetic
python Main.py <instance_file> <iterations> <population_size> <mutation_prob> <local_search_prob>
```

**Example:**
```bash
python Main.py ../Instances/20_5.txt 100 50 0.05 0.05
```

**Parameters:**
- `instance_file`: Path to problem instance
- `iterations`: Number of generations
- `population_size`: Size of population
- `mutation_prob`: Probability of mutation (0.0 - 1.0)
- `local_search_prob`: Probability of applying local search (0.0 - 1.0)

**Batch execution:**
```bash
./execute.sh ../Instances/20_5.txt 100
```

### Iterated Local Search (ILS)

```bash
cd Local_search
python Main.py <instance_file> <iterations>
```

**Example:**
```bash
python Main.py ../Instances/50_10.txt 1000
```

**Parameters:**
- `instance_file`: Path to problem instance
- `iterations`: Number of ILS iterations

**Output includes:**
- Solution permutation
- Objective function value
- Lower bound
- Gap percentage
- Execution time

### Variable Neighborhood Descent (VND)

```bash
cd Neighborhood
python Main.py <instance_file>
```

**Example:**
```bash
python Main.py ../Instances/100_5.txt
```

**Parameters:**
- `instance_file`: Path to problem instance

### GRASP Construction

```bash
cd Random
python Grasp_construction.py <instance_file> <alpha>
```

**Example:**
```bash
python Grasp_construction.py ../Instances/20_10.txt 0.3
```

**Parameters:**
- `instance_file`: Path to problem instance
- `alpha`: Greediness parameter (0.0 = pure greedy, 1.0 = random)

### Generate New Instances

```bash
cd ProblemGen
./ProblemGen <seed> <num_jobs> <num_machines> > output_file.txt
```

**Example:**
```bash
./ProblemGen 12345 30 15 > ../Instances/30_15.txt
```

## Problem Instances

The project includes 9 standard test instances with varying complexity:

| Instance   | Jobs | Machines | Best Known Solution |
|------------|------|----------|---------------------|
| 20_5.txt   | 20   | 5        | 14,033              |
| 20_10.txt  | 20   | 10       | 20,911              |
| 20_20.txt  | 20   | 20       | 33,623              |
| 50_5.txt   | 50   | 5        | 64,980              |
| 50_10.txt  | 50   | 10       | 87,979              |
| 50_20.txt  | 50   | 20       | 126,338             |
| 100_5.txt  | 100  | 5        | 256,061             |
| 100_10.txt | 100  | 10       | 301,453             |
| 100_20.txt | 100  | 20       | 370,035             |

### Instance Format

Instances follow the format:
```
p[0][0] p[0][1] ... p[0][n-1]
p[1][0] p[1][1] ... p[1][n-1]
...
p[m-1][0] p[m-1][1] ... p[m-1][n-1]
```

Where `p[i][j]` is the processing time of job `j` on machine `i`.

## Results

### Algorithm Performance Summary

Based on experimental results with various instances:

**Genetic Algorithm:**
- ✅ Good solution quality for large instances
- ⚠️ Slower execution time
- 🎯 Best for: Large problems where quality matters more than speed

**Iterated Local Search (ILS):**
- ✅ Excellent balance between quality and speed
- ✅ Consistently good results
- 🎯 Best for: General-purpose use

**Variable Neighborhood Descent (VND):**
- ✅ Fast execution
- ✅ Deterministic results
- ⚠️ May get stuck in local optima
- 🎯 Best for: Quick solutions, small to medium instances

**GRASP Construction:**
- ✅ Very fast construction
- ⚠️ Lower solution quality without local search
- 🎯 Best for: Initial solution generation, hybrid approaches

### Quality vs. Time Trade-off

```
Quality:  GA > ILS > VND > GRASP
Speed:    GRASP > VND > ILS > GA
```

## Technical Details

### Cost Calculation

The objective function calculates the total flowtime as follows:

```python
def calc_cost(solution):
    """
    Calculate total flowtime for a given job permutation.

    Args:
        solution: Array representing job permutation

    Returns:
        Total flowtime (sum of completion times)
    """
    return sum([schedule[-1] for schedule in calc_schedule(solution)])
```

### Schedule Computation

For each job in the permutation, the completion time on each machine is calculated considering:
1. Completion time on the previous machine
2. Availability of the current machine

```python
C[i][j] = max(C[i][j-1], C[i-1][j]) + p[i][j]
```

Where:
- `C[i][j]`: Completion time of job `i` on machine `j`
- `p[i][j]`: Processing time of job `i` on machine `j`

### Lower Bound

A simple lower bound is calculated as:

```python
bound = sum(all processing times)
```

This represents the theoretical minimum if all machines could process simultaneously without waiting.

### Neighborhood Structures

#### 1. Swap Neighborhood
Exchanges positions of two jobs in the permutation.

#### 2. Insertion Neighborhood
Removes a job and inserts it in a different position.

#### 3. Best Insertion
Iteratively builds a solution by inserting each unscheduled job at its best position.

## References

This implementation is based on the following research:

1. **Taillard, E.** (1993). *Benchmarks for basic scheduling problems*. European Journal of Operational Research, 64, 278-285.

2. Problem generator based on Taillard's method for generating flowshop instances.

3. Research paper included in `/Article` directory (in Spanish): *"Problema del flujo de trabajos"* by Alejandro Salgado Gómez and Juan Carlos Rivera (2017).

### Key Academic References

- **NP-Hardness**: Garey, M. R., Johnson, D. S., & Sethi, R. (1976). The complexity of flowshop and jobshop scheduling.
- **Genetic Algorithms**: Holland, J. H. (1992). Adaptation in natural and artificial systems.
- **GRASP**: Feo, T. A., & Resende, M. G. (1995). Greedy randomized adaptive search procedures.
- **ILS**: Lourenço, H. R., Martin, O. C., & Stützle, T. (2003). Iterated local search.
- **VND**: Hansen, P., & Mladenović, N. (2001). Variable neighborhood search.
