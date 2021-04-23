#include "preprocessor_kernels.h"
#include "preprocessor.h"
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include "histogram.cu"


//* 1st version of kernel
__global__ void assign_scores_kernel ( int * d_authorized_caldidates_array, 
   int * d_histogram_array, long * d_scores_array, int num_vars ) {
   
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
   int * histogram_array, int * scores_array, int num_vars ) {
   
   int * d_authorized_caldidates_array;
   int * d_histogram_array;
   long * d_scores_array;
   
   int size_of_authorized_caldidates_array = sizeof(int)*(num_vars+1);
   int size_of_histogram_array = sizeof(int)*(2*num_vars+1);
   int size_of_scores_array = sizeof(long)*(num_vars+1);
   
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
   

   cudaEvent_t start_gpu_1, end_gpu_1;
   float msecs_gpu_1;
   cudaEventCreate(&start_gpu_1);
   cudaEventCreate(&end_gpu_1);
   cudaEventRecord(start_gpu_1, 0);
      assign_scores_kernel <<< num_blocks, block_size >>> ( d_authorized_caldidates_array, 
         d_histogram_array, d_scores_array, num_vars );
      cudaDeviceSynchronize();
   cudaEventRecord(end_gpu_1, 0);
   cudaEventSynchronize(end_gpu_1);
   cudaEventElapsedTime(&msecs_gpu_1, start_gpu_1, end_gpu_1);
   cudaEventDestroy(start_gpu_1);
   cudaEventDestroy(end_gpu_1);
   std::cout<<"\nassign_scores_kernel done in "<<msecs_gpu_1<<" milliseconds.\n";


   cudaMemcpy ( (void*) authorized_caldidates_array, 
      (void*) d_authorized_caldidates_array, 
      size_of_authorized_caldidates_array, cudaMemcpyDeviceToHost );
   cudaMemcpy ( (void*) scores_array, 
      (void*) d_scores_array, 
      size_of_scores_array, cudaMemcpyDeviceToHost );

}


void Preprocessor::run_create_histogram_array_kernel() {
  
   cudaEvent_t start_gpu_1, end_gpu_1;
   float msecs_gpu_1;
   cudaEventCreate(&start_gpu_1);
   cudaEventCreate(&end_gpu_1);
   cudaEventRecord(start_gpu_1, 0);
         thrust::device_vector<int> final_histogram;
         final_histogram.resize(2*num_vars+1);      
         for(int i=0; i<cnf->getNumClauses(); i++) {
            Clause c = cnf->getClause(i);
            thrust::device_vector<int> input_array( c.getClauseAsArray(), c.getClauseAsArray() + c.getNumLits());
            thrust::device_vector<int> histogram;
            histogram.resize(2*num_vars+1);
            dense_histogram(input_array, histogram);
            thrust::transform(histogram.begin(), histogram.end(), final_histogram.begin(), final_histogram.begin(), thrust::plus<int>());
         }
         thrust::copy(final_histogram.begin(), final_histogram.end(), histogram_array);
   cudaEventRecord(end_gpu_1, 0);
   cudaEventSynchronize(end_gpu_1);
   cudaEventElapsedTime(&msecs_gpu_1, start_gpu_1, end_gpu_1);
   cudaEventDestroy(start_gpu_1);
   cudaEventDestroy(end_gpu_1);
   std::cout<<"\nrun_create_histogram_array_kernel done in "<<msecs_gpu_1<<" milliseconds.\n";

}

void Preprocessor::run_sort_wrt_scores_kernel(){
   cudaEvent_t start_gpu_1, end_gpu_1;
   float msecs_gpu_1;
   cudaEventCreate(&start_gpu_1);
   cudaEventCreate(&end_gpu_1);
   cudaEventRecord(start_gpu_1, 0);
         thrust::sort_by_key(scores_array + 1, scores_array + num_vars + 1, authorized_caldidates_array + 1);
   cudaEventRecord(end_gpu_1, 0);
   cudaEventSynchronize(end_gpu_1);
   cudaEventElapsedTime(&msecs_gpu_1, start_gpu_1, end_gpu_1);
   cudaEventDestroy(start_gpu_1);
   cudaEventDestroy(end_gpu_1);
   std::cout<<"\nrun_sort_wrt_scores_kernel done in "<<msecs_gpu_1<<" milliseconds.\n";
}