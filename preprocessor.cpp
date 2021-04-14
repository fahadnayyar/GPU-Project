#include "preprocessor.h"
#include "preprocessor_kernels.h"
#include <bits/stdc++.h>
using namespace std;

extern "C" void run_assign_scores_kernel ( int * authorized_caldidates_array, 

int * histogram_array, 
int * scores_array,
int num_vars ); 

void Preprocessor::do_parallel_preprocessing() {
   constructAGPU();  
   createOccurTable();
   LCVE_algorithm();
}

// * Algorithm 1
void Preprocessor::constructAGPU() {
      
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
   //* CPU sequrntial histogram computation.
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
   cout << "DEBUGGING: initialize_hisotgram_array finished.\n" << endl;
   print_histogram_array();
}

void Preprocessor::assign_scores() {
   //* TODO: imp: initialize authorized_caldidates_array
   authorized_caldidates_array = new int[num_vars+1];
   scores_array = new int[num_vars+1];
   
   //* executiig parallel assign_scores_kernel
   run_assign_scores_kernel ( authorized_caldidates_array, histogram_array, scores_array, num_vars );
   cout << "DEBUGGING: run_assign_scores_kernel finished.\n" << endl;
   print_authorized_caldidates_array();
   print_scores_array();
}

bool compareScores(pair<int, int> i1, pair<int, int> i2)
{
    return (i1.second < i2.second);
}

void Preprocessor::sort_authorized_caldidates_according_to_scores() {
   pair<int, char> pairt[num_vars];

   for (int i = 1; i < num_vars + 1; i++) 
   {
      pairt[i-1].first = authorized_caldidates_array[i];
      pairt[i-1].second = scores_array[i];
   }

   // Sorting the pair array.
   sort(pairt, pairt + num_vars, compareScores);

   // Modifying original arrays
   for (int i = 1; i < num_vars + 1; i++) 
   {
      authorized_caldidates_array[i] = pairt[i-1].first;
      scores_array[i] = pairt[i-1].second;
   }
   cout << "DEBUGGING: sort_authorized_caldidates_according_to_scores finished.\n" << endl;
   print_authorized_caldidates_array();
   print_scores_array();

}

void Preprocessor::prune() {
   int x = mu;
   int * elem = upper_bound(scores_array + 1, scores_array + num_vars + 1, x);
   cout << "* elem: " << *elem << "\n";
   cutoffpoint = (((int)(elem - scores_array)));
   cout << "DEBUGGING: prune finished.\n" << endl;
   cout << "DEBUGGING: cutoffpoint: " << cutoffpoint << "\n" << endl;
}


//* algorithm 2
void Preprocessor::LCVE_algorithm() {
   bool * frozen_array = new bool[num_vars];
   for (int i=0; i< num_vars; i++) {
      frozen_array[i] = false;
   }
   for (int x=1; x < cutoffpoint; x++) {
      if ( ! frozen_array[x] ) {
         append_electd_candidate(x);
         int index_p = 2*x;
         int index_n = 2*x-1;
         OccurList& occur_list_p = occur_table->getOccurList(index_p);
         OccurList& occur_list_n = occur_table->getOccurList(index_n);
        
        
        

         for (int i=0 ;i<occur_list_p.getOccurListSize(); i++){
            int clause_index = occur_list_p.getClauseIndex(i);
            Clause& c = cnf->getClause(clause_index);     
            for (int j=0; j<c.getNumLits(); j++) {
               int v = c.getLit(j)/2;
               if (v != x) {
                  //* TODO: optimize "v belongs to" A
                  int flag = 0;
                  for (int k=0;k<cutoffpoint;k++) {
                     if (authorized_caldidates_array[k] == v){
                        flag = 1;
                        break;
                     }
                  }
                  if (flag == 1) {
                     frozen_array[v] = true;
                  }
               }
            }
         }
         
         


         for (int i=0 ;i<occur_list_n.getOccurListSize(); i++) {
            int clause_index = occur_list_p.getClauseIndex(i);
            Clause& c = cnf->getClause(clause_index);     
            for (int j=0; j<c.getNumLits(); j++) {
               int v = c.getLit(j)/2;
               if (v != x) {
                  //* TODO: optimize "v belongs to" A
                  int flag = 0;
                  for (int k=0;k<cutoffpoint;k++) {
                     if (authorized_caldidates_array[k] == v){
                        flag = 1;
                        break;
                     }
                  }
                  if (flag == 1) {
                     frozen_array[v] = true;
                  }
               }
            }   
         }




      }
   }

   cout << "DEBUGGING: LCVE_algorithm finished  \n\n";
   print_elected_candidates_vector();

}


void Preprocessor::createOccurTable() {
   occur_table = new OccurTab();
   occur_table->setNumLits(2*num_vars);
   occur_table->initializeArray();
   for (int i=1; i<((2*num_vars)+1); i++ ) {
      OccurList& occur_list = occur_table->getOccurList(i);
      occur_list.setOccurListSize(histogram_array[i]);
      occur_list.initializeArray();
      // cout << "i: " << i << "\n";
      // cout << "occur_list_size : " << occur_list.getOccurListSize() << "\n";
   }
   for (int i=0; i<num_clauses; i++) {
      Clause& c = cnf->getClause(i);
      for (int j=0; j< c.getNumLits(); j++) {
         int lit = c.getLit(j);
         // cout << "HEHE: lit :" << lit << "\n";
         occur_table->getOccurList(lit).addClause(i);
      }
   }
   cout << "DEBUGGING: OccurTable created  \n\n";
   print_occur_table();

}


void Clause::print_clause() {
   for(int i=0; i< num_lits; i++){
      cout << literal_array[i] << " ";
   }
   cout << "\n\n";
}

void Preprocessor::print_authorized_caldidates_array() {
   cout << "DEBUGGING: authorized_caldidates_array: \n";
   for (int i=1; i<num_vars+1; i++) {
      cout << authorized_caldidates_array[i] << ", \n";
   }
   cout << endl;
}

void Preprocessor::print_scores_array() {
   cout << "DEBUGGING: scores_array: \n";
   for (int i=1; i<num_vars+1; i++) {
      cout << scores_array[i] << ", \n";
   }
   cout << endl;
}

void Preprocessor::print_histogram_array() {
   cout << "DEBUGGING: histogram_array: \n";
   for (int i=1; i<2*num_vars+1; i++) {
      cout << histogram_array[i] << ", \n";
   }
   cout << endl;
}

void Preprocessor::print_occur_table() {
   cout << "DEBUGGING: occur_table: \n";
   for (int i=1; i<2*num_vars+1; i++) {
      OccurList occur_list =  occur_table->getOccurList(i);
      cout << "Printing occur_list for Literal: " << i << "\n";
      for (int j=0; j<occur_list.getOccurListSize(); j++) {
         cout << occur_list.getClauseIndex(j) << ", \n";
      }
   }
   cout << endl;
}

void Preprocessor::print_elected_candidates_vector() {
   cout << "DEBUGGING: elected_candidates_vector: \n";
   for (auto& it : elected_candidates_vector) {
      cout << it << ", \n";
    }
}