# UCS645-Assignments




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
