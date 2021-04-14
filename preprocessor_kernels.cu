#include "preprocessor_kernels.h"
#include <cmath>

//* 1st version of kernel
__global__ void assign_scores_kernel ( int * d_authorized_caldidates_array, 
   int * d_histogram_array, int * d_scores_array, int num_vars ) 
{
   
   int tid = threadIdx.x + blockDim.x*blockIdx.x; 
   int stride = gridDim.x*blockDim.x;

   while ( tid < num_vars ) {
      
      int x = tid+1;
      d_authorized_caldidates_array[x] = x;
      int h_x_p = d_histogram_array[2*x];
      int h_x_n = d_histogram_array[2*x-1];
      if (  h_x_p == 0 || h_x_n == 0 ) {
         d_scores_array[x] = max(h_x_p, h_x_n);
      } else {
         d_scores_array[x] = h_x_p * h_x_n;
      }
      tid = tid + stride;

   }

}

extern "C" void run_assign_scores_kernel ( int * authorized_caldidates_array, 
   int * histogram_array, int * scores_array, int num_vars ) 
{
   
   int * d_authorized_caldidates_array;
   int * d_histogram_array;
   int * d_scores_array;
   
   int size_of_authorized_caldidates_array = sizeof(int)*(num_vars+1);
   int size_of_histogram_array = sizeof(int)*(2*num_vars+1);
   int size_of_scores_array = sizeof(int)*(num_vars+1);
   
   cudaMalloc ( ( void ** ) &d_authorized_caldidates_array, 
      size_of_authorized_caldidates_array );
   cudaMalloc ( ( void ** ) &d_histogram_array, 
      size_of_histogram_array );
   cudaMalloc ( ( void ** ) &d_scores_array, 
      size_of_scores_array );
   
   cudaMemcpy ( (void*) d_histogram_array, 
      (void*) histogram_array, size_of_histogram_array, 
      cudaMemcpyHostToDevice );

   //* TODO: fix num_blocks, it is too huge right now (millions of vars are possible)
   int num_blocks = ceil( ((double)num_vars) / ((double)256));
   int block_size = 256;
   assign_scores_kernel <<< num_blocks, block_size >>> ( d_authorized_caldidates_array, 
      d_histogram_array, d_scores_array, num_vars );

   cudaDeviceSynchronize();

   cudaMemcpy ( (void*) authorized_caldidates_array, 
      (void*) d_authorized_caldidates_array, 
      size_of_authorized_caldidates_array, cudaMemcpyDeviceToHost );
   cudaMemcpy ( (void*) scores_array, 
      (void*) d_scores_array, 
      size_of_scores_array, cudaMemcpyDeviceToHost );

}