# UCS645-Assignments 1,2,3,4,5




Assignment 2: MPI

This repository contains MPI-based implementations of fundamental parallel computing algorithms. Below are the tasks implemented:

1. Estimate the value of Pi using the Monte Carlo method

Demonstrates basic MPI functions.

Uses parallel random sampling to approximate Pi.

2. Matrix Multiplication using MPI

Computes the multiplication of a 70×70 matrix.

Compares serial and parallel execution times using omp_get_wtime().

3. Parallel Sorting using MPI (Odd-Even Sort)

Implements Odd-Even sort in parallel using MPI.

4. Heat Distribution Simulation using MPI

Simulates heat flow over a 2D grid using MPI-based computation.

5. Parallel Reduction using MPI

Performs parallel summation of elements using MPI Reduce.

6. Parallel Dot Product using MPI

Computes the dot product of two vectors using MPI.

7. Parallel Prefix Sum (Scan) using MPI

Implements prefix sum computation using MPI-based parallelism.

8. Parallel Matrix Transposition using MPI

Performs matrix transposition using MPI communication.




Assignment 3: Advanced MPI Programs


Q1. DAXPY Loop

Implements the DAXPY operation: X[i] = a*X[i] + Y[i].

Uses MPI to parallelize the computation.

Measures speedup compared to a serial execution.


Q2. Calculation of π - MPI Bcast and MPI Reduce

Parallel implementation of Pi calculation using integration.

Uses MPI_Bcast to distribute num_steps across processes.

Uses MPI_Reduce to accumulate results from all processes.


Q3. Parallel Prime Number Finder using MPI

Finds all prime numbers up to a given maximum using MPI.

Master process distributes numbers for checking to worker processes.

Worker processes test numbers and communicate results using MPI_Send and MPI_Recv.




Assignment 4:


Overview This project implements a basic vector addition using NVIDIA's CUDA framework. It demonstrates:

Vector Addition: Adding two vectors element-wise on the GPU.

Statically Defined Global Variables: Utilizing statically allocated memory on the device.

Kernel Execution Timing: Measuring the execution time of the CUDA kernel.

Memory Bandwidth Calculation: Computing both theoretical and measured memory bandwidths.​University of Notre Dame +3 QuantStart +3 GitHub +3

Prerequisites Hardware: NVIDIA GPU with CUDA support.

Software:

CUDA Toolkit installed.

C++ compiler compatible with CUDA (e.g., nvcc).​vrushankdes.ai +5 Learn PDC +5 QuantStart +5

Compilation Compile the CUDA program using the NVIDIA CUDA Compiler:

bash Copy Edit nvcc -O3 -o vector_add vector_add.cu Execution Run the compiled executable:

bash Copy Edit ./vector_add



This project comprises three CUDA programming exercises:

Sum of First n Integers:

Iterative Approach: Calculates the sum of the first n integers using a loop.

Formula-Based Approach: Calculates the sum using the direct formula.

Vector Addition:

Implements vector addition using statically defined global variables.

Records kernel execution time.

Calculates theoretical and measured memory bandwidth.

Prerequisites NVIDIA GPU with CUDA support.

CUDA Toolkit installed.

C++ compiler compatible with CUDA.

Compilation Use the NVIDIA CUDA Compiler (nvcc) to compile the program:

bash Copy Edit nvcc -O3 -o cuda_exercises cuda_exercises.cu Execution Run the compiled executable:

bash Copy Edit ./cuda_exercises Exercises

Sum of First n Integers a. Iterative Approach Description: Calculates the sum by iterating from 1 to n.
Implementation: Each thread computes the sum independently.

b. Formula-Based Approach Description: Uses the formula sum = n * (n + 1) / 2.

Implementation: Each thread applies the formula directly.

Vector Addition Description: Adds two vectors of size N element-wise.
Implementation:

Uses statically defined global device arrays.

Launches a kernel where each thread computes one element of the result vector.

Measures kernel execution time using CUDA events.

Calculates:

Theoretical Memory Bandwidth: Based on device properties.

Measured Memory Bandwidth: Based on actual data transfer and execution time.

Sample Output yaml Copy Edit Vector addition successful. Kernel execution time: 0.123456 ms Theoretical Memory Bandwidth: 320.00 GB/s Measured Memory Bandwidth: 150.00 GB/s Note: Actual output will vary based on hardware and system load.

Profiling To profile the application using nvprof:

bash Copy Edit nvprof ./cuda_exercises
