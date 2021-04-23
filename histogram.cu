// REFERENCE: https://github.com/NVIDIA/thrust/blob/master/examples/histogram.cu
// Code for histogram is copied from above reference.

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

// dense histogram using binary search
template <typename Vector1, 
          typename Vector2>
void dense_histogram(Vector1& input, Vector2& histogram)
{
  typedef typename Vector1::value_type ValueType; // input value type
  typedef typename Vector2::value_type IndexType; // histogram index type

  // copy input data (could be skipped if input is allowed to be modified)
  thrust::device_vector<ValueType> data(input);
    
  // print the initial data
//   print_vector("initial data", data);

  // sort data to bring equal elements together
//   thrust::sort(data.begin(), data.end());
  
  // print the sorted data
//   print_vector("sorted data", data);

  // number of histogram bins is equal to the maximum value plus one
  IndexType num_bins = data.back() + 1;

  // resize histogram storage
  histogram.resize(num_bins);
  
  // find the end of each bin of values
  thrust::counting_iterator<IndexType> search_begin(0);
  thrust::upper_bound(data.begin(), data.end(),
                      search_begin, search_begin + num_bins,
                      histogram.begin());
  
  // print the cumulative histogram
//   print_vector("cumulative histogram", histogram);

  // compute the histogram by taking differences of the cumulative histogram
  thrust::adjacent_difference(histogram.begin(), histogram.end(),
                              histogram.begin());

  // print the histogram
//   print_vector("histogram", histogram);
}