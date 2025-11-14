# Parallel Programming Examples

A comprehensive collection of parallel and concurrent programming examples in C/C++, demonstrating various paradigms and techniques for parallel computation.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Compilation and Execution](#compilation-and-execution)
- [Examples](#examples)
  - [Fork (Process-Based Parallelism)](#fork-process-based-parallelism)
  - [POSIX Threads](#posix-threads)
  - [OpenMP](#openmp)
  - [MPI (Message Passing Interface)](#mpi-message-passing-interface)
- [Key Concepts](#key-concepts)
- [Learning Path](#learning-path)
- [Resources](#resources)

## Overview

This repository contains practical examples of parallel programming using four main approaches:

1. **Fork** - Process-based parallelism using Unix `fork()` system call
2. **POSIX Threads** - Shared memory multithreading
3. **OpenMP** - High-level shared memory parallelism
4. **MPI** - Distributed memory parallelism for cluster computing

Each directory contains self-contained examples with Makefiles for easy compilation.

## Prerequisites

### Software Requirements

- **GCC/G++** compiler with C++11 support
- **POSIX Threads** library (usually included with Unix-like systems)
- **OpenMP** support (included in modern GCC versions)
- **MPI** implementation (OpenMPI or MPICH)

### Installing Dependencies

#### macOS
```bash
# Install GCC with OpenMP support
brew install gcc

# Install OpenMPI
brew install open-mpi
```

#### Ubuntu/Debian
```bash
# Install build essentials
sudo apt-get update
sudo apt-get install build-essential

# Install OpenMPI
sudo apt-get install libopenmpi-dev openmpi-bin
```

#### Fedora/RHEL
```bash
# Install development tools
sudo dnf groupinstall "Development Tools"

# Install OpenMPI
sudo dnf install openmpi openmpi-devel
```

## Project Structure

```
Parallel/
├── Fork/                    # Process-based parallelism examples
│   ├── Basic/              # Simple fork() usage
│   ├── Var/                # Variable behavior in parent/child
│   ├── Execv/              # Process replacement with execv()
│   └── File/               # Shared file descriptors
├── Threads/                # POSIX Threads examples
│   ├── Hello/              # Basic thread creation
│   ├── Join/               # Thread joining and return values
│   ├── Mutex/              # Mutex synchronization
│   ├── Semaphores/         # Semaphore synchronization
│   └── CondVar/            # Condition variables
├── OpenMp/                 # OpenMP examples
│   ├── NumThreads/         # Basic parallel regions
│   └── For/                # Parallel for loops
└── Mpi/                    # MPI examples
    ├── Initialization/     # MPI setup
    ├── ListProcess/        # Master-worker pattern
    ├── SendReceive/        # Point-to-point communication
    ├── SendReceiveDistributed/  # Distributed communication
    └── Distributed/        # Distributed computing
```

## Compilation and Execution

Each example directory contains a `Makefile`. To compile and run:

```bash
# Navigate to any example directory
cd Fork/Basic

# Compile
make

# Run (name may vary by example)
./Basic
```

To clean build artifacts:
```bash
make clean
```

## Examples

### Fork (Process-Based Parallelism)

Process-based parallelism using Unix `fork()` system call. Each forked process has its own memory space.

#### Basic
**Location**: `Fork/Basic/`

Demonstrates the simplest use of `fork()` to create a child process.

```bash
cd Fork/Basic && make && ./Basic
```

**Key Concepts**:
- Process creation with `fork()`
- Parent and child process identification
- Return value of `fork()` (0 for child, PID for parent)

#### Var
**Location**: `Fork/Var/`

Illustrates how variables are copied between parent and child processes.

```bash
cd Fork/Var && make && ./Var
```

**Key Concepts**:
- Memory isolation between processes
- Copy-on-write semantics
- Independent variable modifications

**Output Explanation**: Changes in child process variables don't affect parent process variables and vice versa.

#### Execv
**Location**: `Fork/Execv/`

Shows how to replace a child process with a different program using `execv()`.

```bash
cd Fork/Execv && make && ./Execv
```

**Key Concepts**:
- Process replacement with `execv()`
- Command-line arguments passing
- The `exec` family of functions

**Use Case**: Creating new processes that run different programs (like shells do).

#### File
**Location**: `Fork/File/`

Demonstrates file descriptor sharing between parent and child processes.

```bash
cd Fork/File && make && ./File
```

**Key Concepts**:
- File descriptor inheritance
- Process synchronization with `wait()`
- Shared file offset between processes

**Output**: Creates `file.txt` with content written by both parent and child.

---

### POSIX Threads

Thread-based parallelism using POSIX threads (pthreads). Threads share the same memory space.

#### Hello
**Location**: `Threads/Hello/`

Basic thread creation and execution.

```bash
cd Threads/Hello && make && ./Hello
```

**Key Concepts**:
- Thread creation with `pthread_create()`
- Passing arguments to threads
- Thread termination with `pthread_exit()`

**Note**: Demonstrates potential race conditions in output.

#### Join
**Location**: `Threads/Join/`

Shows how to wait for threads to complete and retrieve return values.

```bash
cd Threads/Join && make && ./Join
```

**Key Concepts**:
- Thread synchronization with `pthread_join()`
- Retrieving thread return values
- Timing differences in thread execution

**Use Case**: Ensures main thread waits for worker threads to complete.

#### Mutex
**Location**: `Threads/Mutex/`

Demonstrates mutual exclusion using mutexes to protect shared data.

```bash
cd Threads/Mutex && make && ./Mutex
```

**Key Concepts**:
- Critical sections
- Mutex locking with `pthread_mutex_lock()`
- Mutex unlocking with `pthread_mutex_unlock()`
- Race condition prevention

**Use Case**: Protecting shared variables from concurrent modification.

#### Mutex/Without
**Location**: `Threads/Mutex/Without/`

Shows the problems that occur without proper synchronization.

```bash
cd Threads/Mutex/Without && make && ./Without
```

**Key Concepts**:
- Race conditions
- Data corruption without synchronization
- Importance of critical sections

**Comparison**: Run this alongside the Mutex example to see the difference.

#### Semaphores
**Location**: `Threads/Semaphores/`

Implements synchronization using semaphores instead of mutexes.

```bash
cd Threads/Semaphores && make && ./Semaphores
```

**Key Concepts**:
- Semaphore initialization with `sem_init()`
- Waiting with `sem_wait()`
- Signaling with `sem_post()`
- Binary semaphores as mutex alternatives

**Comparison**: Functionally similar to mutex example but uses semaphores.

#### CondVar
**Location**: `Threads/CondVar/`

Demonstrates condition variables for thread coordination.

```bash
cd Threads/CondVar && make && ./CondVar
```

**Key Concepts**:
- Condition variable waiting with `pthread_cond_wait()`
- Signaling with `pthread_cond_signal()`
- Must be used with a mutex
- Event-based synchronization

**Use Case**: One thread waits for a condition to be met by other threads.

---

### OpenMP

High-level shared memory parallelism using compiler directives.

#### NumThreads
**Location**: `OpenMp/NumThreads/`

Basic parallel region creation.

```bash
cd OpenMp/NumThreads && make && ./NumThreads
```

**Key Concepts**:
- `#pragma omp parallel` directive
- Thread identification with `omp_get_thread_num()`
- Automatic thread management

**Note**: Number of threads is determined by system or `OMP_NUM_THREADS` environment variable.

#### For
**Location**: `OpenMp/For/`

Parallel loop execution with OpenMP.

```bash
# Set number of threads
export OMP_NUM_THREADS=4

cd OpenMp/For && make && ./For
```

**Key Concepts**:
- `#pragma omp parallel for` directive
- Automatic loop iteration distribution
- Environment variable control (`OMP_NUM_THREADS`)
- Load balancing across threads

**Use Case**: Parallelizing independent loop iterations for performance.

---

### MPI (Message Passing Interface)

Distributed memory parallelism for multi-node cluster computing.

#### Initialization
**Location**: `Mpi/Initialization/`

Basic MPI program structure.

```bash
cd Mpi/Initialization && make

# Run with 4 processes
mpirun -np 4 ./Initialization
```

**Key Concepts**:
- MPI initialization with `MPI_Init()`
- MPI finalization with `MPI_Finalize()`
- MPI program structure

**Note**: This is the minimal MPI program template.

#### ListProcess
**Location**: `Mpi/ListProcess/`

Master-worker pattern where workers send messages to master.

```bash
cd Mpi/ListProcess && make

# Run with multiple processes
mpirun -np 5 ./ListProcess
```

**Key Concepts**:
- Process rank with `MPI_Comm_rank()`
- Process count with `MPI_Comm_size()`
- Master-worker communication pattern
- Sequential message collection

**Use Case**: Coordinating multiple processes with a central master.

#### SendReceive
**Location**: `Mpi/SendReceive/`

Point-to-point communication between two processes.

```bash
cd Mpi/SendReceive && make

# Run with at least 2 processes
mpirun -np 2 ./SendReceive
```

**Key Concepts**:
- Sending data with `MPI_Send()`
- Receiving data with `MPI_Recv()`
- Message tags
- MPI data types (e.g., `MPI_INT`)
- Blocking communication

**Use Case**: Direct data exchange between specific processes.

#### SendReceiveDistributed
**Location**: `Mpi/SendReceiveDistributed/`

Point-to-point communication across multiple machines.

```bash
cd Mpi/SendReceiveDistributed && make

# Run on distributed nodes (requires machine file)
mpirun -np 2 --hostfile machines ./SendReceiveDistributed
```

**Key Concepts**:
- Multi-machine MPI execution
- Host file configuration
- Network-based communication
- Distributed memory model

**Setup**: Requires SSH access to machines listed in `machines` file.

#### Distributed
**Location**: `Mpi/Distributed/`

Distributed computation across cluster nodes.

```bash
cd Mpi/Distributed && make

# Run across cluster
mpirun -np 4 --hostfile machines ./Distributed
```

**Key Concepts**:
- Cluster computing
- Network latency considerations
- Distributed memory architecture

**Setup**: Requires properly configured MPI cluster with `machines` file.

---

## Key Concepts

### Concurrency vs Parallelism

- **Concurrency**: Multiple tasks making progress (may or may not run simultaneously)
- **Parallelism**: Multiple tasks executing simultaneously on multiple cores

### Shared Memory vs Distributed Memory

| Aspect | Shared Memory (Threads, OpenMP) | Distributed Memory (Processes, MPI) |
|--------|----------------------------------|-------------------------------------|
| Memory Space | Shared | Separate |
| Communication | Direct memory access | Message passing |
| Synchronization | Mutexes, semaphores, barriers | Send/Receive operations |
| Overhead | Lower | Higher |
| Scalability | Limited to single machine | Scales across clusters |

### Synchronization Primitives

1. **Mutex**: Mutual exclusion lock for protecting critical sections
2. **Semaphore**: Generalized counter for resource access control
3. **Condition Variable**: Event-based coordination between threads
4. **Barrier**: Synchronization point where all threads must wait
5. **Message Passing**: Explicit send/receive for process coordination

### Race Conditions

A race condition occurs when:
- Multiple threads/processes access shared data
- At least one modifies the data
- No proper synchronization is used

**Solution**: Use mutexes, semaphores, or message passing to coordinate access.

### Deadlock

Deadlock occurs when two or more threads/processes wait for each other indefinitely.

**Four conditions for deadlock**:
1. Mutual exclusion
2. Hold and wait
3. No preemption
4. Circular wait

**Prevention**: Acquire locks in consistent order, use timeouts, avoid nested locks.

---

## Learning Path

### Beginner Path

1. **Fork/Basic** - Understand process creation
2. **Fork/Var** - Learn about memory isolation
3. **Threads/Hello** - Basic thread creation
4. **Threads/Join** - Thread synchronization
5. **OpenMp/NumThreads** - High-level parallelism

### Intermediate Path

6. **Threads/Mutex** - Synchronization primitives
7. **Threads/Mutex/Without** - Understand race conditions
8. **Threads/Semaphores** - Alternative synchronization
9. **OpenMp/For** - Parallel loops
10. **Mpi/Initialization** - Distributed computing basics

### Advanced Path

11. **Threads/CondVar** - Complex synchronization
12. **Fork/Execv** - Process replacement
13. **Fork/File** - Shared resources
14. **Mpi/SendReceive** - Message passing
15. **Mpi/ListProcess** - Master-worker patterns
16. **Mpi/Distributed** - Cluster computing

---

## Performance Considerations

### When to Use Each Approach

#### Use Fork When:
- Need strong isolation between tasks
- Want to run different programs
- Security/stability is critical (crash isolation)

#### Use Threads When:
- Tasks need to share data frequently
- Lower overhead is required
- Working on a single machine
- Need fine-grained parallelism

#### Use OpenMP When:
- Parallelizing existing sequential code
- Loop-based parallelism
- Want simple, directive-based approach
- Shared memory is sufficient

#### Use MPI When:
- Need to scale beyond one machine
- Have large distributed datasets
- Computing on clusters/supercomputers
- Explicitly manage data distribution

### Common Pitfalls

1. **Race Conditions**: Always protect shared data
2. **Deadlock**: Acquire locks in consistent order
3. **False Sharing**: Cache line contention in threads
4. **Load Imbalance**: Uneven work distribution
5. **Synchronization Overhead**: Too much synchronization hurts performance
6. **Memory Bandwidth**: Multiple threads competing for memory

---

## Debugging Tips

### For Threads
```bash
# Run with thread sanitizer (GCC)
g++ -fsanitize=thread -g program.cpp -lpthread

# GDB with threads
gdb ./program
(gdb) info threads
(gdb) thread 2  # Switch to thread 2
```

### For MPI
```bash
# Run with verbose output
mpirun -v -np 4 ./program

# Attach GDB to MPI process
mpirun -np 2 xterm -e gdb ./program
```

### For OpenMP
```bash
# Enable runtime debugging
export OMP_DISPLAY_ENV=TRUE
export OMP_STACKSIZE=4M

# Check thread binding
export OMP_DISPLAY_AFFINITY=TRUE
```

---

## Resources

### Books
- "Programming with POSIX Threads" by David R. Butenhof
- "Using OpenMP" by Barbara Chapman et al.
- "MPI: The Complete Reference" by Marc Snir et al.
- "The Art of Multiprocessor Programming" by Maurice Herlihy

### Online Resources
- [POSIX Threads Tutorial](https://computing.llnl.gov/tutorials/pthreads/)
- [OpenMP Official Site](https://www.openmp.org/)
- [MPI Tutorial](https://mpitutorial.com/)
- [Open MPI Documentation](https://www.open-mpi.org/doc/)

### Man Pages
```bash
man pthread_create    # POSIX Threads
man sem_init          # Semaphores
man fork              # Process creation
man mpirun            # MPI execution
```

---

## Troubleshooting

### Common Issues

#### "undefined reference to pthread_create"
**Solution**: Link with `-lpthread` flag
```bash
g++ program.cpp -lpthread
```

#### "undefined reference to omp_get_thread_num"
**Solution**: Add OpenMP flag
```bash
g++ program.cpp -fopenmp
```

#### "mpirun: command not found"
**Solution**: Install MPI and ensure it's in PATH
```bash
# macOS
brew install open-mpi

# Linux
sudo apt-get install openmpi-bin libopenmpi-dev
```

#### MPI programs hang
**Solution**: Check if you're running with enough processes
```bash
# If program expects 2 processes, run with at least 2
mpirun -np 2 ./program
```

#### "Transport endpoint is not connected" (MPI)
**Solution**: Check network connectivity and machine file configuration

---

## Quick Reference

### Compilation Commands

```bash
# Standard C++
g++ -o program program.cpp

# With threads
g++ -o program program.cpp -lpthread

# With OpenMP
g++ -o program program.cpp -fopenmp

# With MPI
mpic++ -o program program.cpp
```

### Execution Commands

```bash
# Standard program
./program

# MPI program (4 processes)
mpirun -np 4 ./program

# MPI distributed (with hostfile)
mpirun -np 8 --hostfile machines ./program

# With environment variables
OMP_NUM_THREADS=8 ./program
```
