//q1


#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void compute_sums(int *iterative_sum, int *formula_sum) {
    int tid = threadIdx.x;

    if (tid == 0) {
        
        int sum = 0;
        for (int i = 1; i <= N; ++i) {
            sum += i;
        }
        *iterative_sum = sum;
    } else if (tid == 1) {
        
        *formula_sum = N * (N + 1) / 2;
    }
}

int main() {
    int h_iterative_sum = 0;
    int h_formula_sum = 0;
    int *d_iterative_sum, *d_formula_sum;

  
    cudaMalloc((void**)&d_iterative_sum, sizeof(int));
    cudaMalloc((void**)&d_formula_sum, sizeof(int));


    compute_sums<<<1, 2>>>(d_iterative_sum, d_formula_sum);

   
    cudaMemcpy(&h_iterative_sum, d_iterative_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_formula_sum, d_formula_sum, sizeof(int), cudaMemcpyDeviceToHost);

    
    printf("Sum using iterative approach: %d\n", h_iterative_sum);
    printf("Sum using formula approach: %d\n", h_formula_sum);

    
    cudaFree(d_iterative_sum);
    cudaFree(d_formula_sum);

    return 0;
}




//q2

#include <iostream>
#include <omp.h>
#include <vector>

void merge(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> leftSub(arr.begin() + left, arr.begin() + mid + 1);
    std::vector<int> rightSub(arr.begin() + mid + 1, arr.begin() + right + 1);

    int i = 0, j = 0, k = left;
    while (i < leftSub.size() && j < rightSub.size()) {
        arr[k++] = (leftSub[i] <= rightSub[j]) ? leftSub[i++] : rightSub[j++];
    }
    while (i < leftSub.size()) arr[k++] = leftSub[i++];
    while (j < rightSub.size()) arr[k++] = rightSub[j++];
}

void parallelMergeSort(std::vector<int>& arr, int left, int right, int depth = 0) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        if (depth < 4) { 
            #pragma omp parallel sections
            {
                #pragma omp section
                parallelMergeSort(arr, left, mid, depth + 1);
                #pragma omp section
                parallelMergeSort(arr, mid + 1, right, depth + 1);
            }
        } else {
            parallelMergeSort(arr, left, mid, depth + 1);
            parallelMergeSort(arr, mid + 1, right, depth + 1);
        }

        merge(arr, left, mid, right);
    }
}

int main() {
    std::vector<int> data(1000);
    
    for (int i = 0; i < 1000; ++i) data[i] = rand() % 10000;

    double start = omp_get_wtime();
    parallelMergeSort(data, 0, data.size() - 1);
    double end = omp_get_wtime();

    std::cout << "Pipelined Parallel Merge Sort Time: " << (end - start) << " seconds\n";
    return 0;
}