#include "preprocessor.h"
#include "preprocessor_kernels.h"

extern "C" void run_assign_scores_kernel ( int * authorized_caldidates_array, 
int * histogram_array, 
int * scores_array ); 

// * Algorithm 1
void Preprocessor::constructAGPU() {
   
//* TODO: imp: initialize authorized_caldidates_array
authorized_caldidates_array = new int[num_vars];

//* 1
   initialize_hisotgram_array();

   //* 2
   assign_scores();

   //* 3
   sort_authorized_caldidates_according_to_scores();

   //* 4
   prune();

}

void Preprocessor::initialize_hisotgram_array() {
   
   histogram_array = new int[2*num_vars+1];
   for(int j=0; j<2*num_vars+1; j++) {
        histogram_array[j]=0;
   }
   for(int i=0; i<cnf->getNumClauses(); i++) {
       Clause c = cnf->getClause(i);
       for(int j=0; j<c.getNumLits(); j++) {
           histogram_array[c.getLit(j)]++;
       }
   }

}

void Preprocessor::assign_scores () {
   run_assign_scores_kernel ( authorized_caldidates_array, histogram_array, scores_array );
}

void Preprocessor::sort_authorized_caldidates_according_to_scores() {

}

void Preprocessor::prune() {

}

void Clause::print_clause() {
      for(int i=0; i< num_lits; i++){
         cout << literal_array[i] << " ";
      }
      cout << "\n";
}