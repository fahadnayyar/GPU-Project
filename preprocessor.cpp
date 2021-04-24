/* @author: Krishna Kariya, Fahad Nayyar, 2021
*/

#include "preprocessor.h"
// #include "preprocessor_kernels.h"
#include <unistd.h>
#include <stdio.h>
#include <limits.h>
#include <bits/stdc++.h>
using namespace std;

extern "C" void run_assign_scores_kernel ( int * authorized_caldidates_array, 

int * histogram_array, 
long * scores_array,
int num_vars ); 

void Preprocessor::do_parallel_preprocessing() {
   constructAGPU();  
   createOccurTable();
   LCVE_algorithm();
   BVIPE_algorithm();
   printCnfInDimacsFile();
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
   //* CPU sequential histogram computation.
   histogram_array = new int[2*num_vars+1];

   // if (mode==0) {
      for(int j=0; j<2*num_vars+1; j++) {
         histogram_array[j]=0;
      }
      for(int i=0; i<cnf->getNumClauses(); i++) {
         Clause c = cnf->getClause(i);
         for(int j=0; j<c.getNumLits(); j++) {
            histogram_array[c.getLit(j)]++;
         }
      }
      
   // } else if (mode==1) { //* GPU parallel histogram computation.
   //    run_create_histogram_array_kernel();
   // } else {
   //    cout << "ERROR: invalid mode in initialize_hisotgram_array()\n";
   //    exit(0);
   // }
   // #ifdef DEBUG
   //    cout << "DEBUGGING: initialize_hisotgram_array finished.\n" << endl;
   //    print_histogram_array();
   // #endif
   
}

void Preprocessor::assign_scores() {
   
   //* initializing authorized_caldidates_array and scores_array
   authorized_caldidates_array = new int[num_vars+1];
   scores_array = new long[num_vars+1];
   
   if (mode==1) {
      //* executiig GPU parallel assign_scores_kernel
      run_assign_scores_kernel ( authorized_caldidates_array, histogram_array, scores_array, num_vars );
   } else if (mode==0) { //* GPU parallel histogram computation.
      //* TODO: complete.
      for (int i=0; i<num_vars; i++) {
         int x = i+1;
         authorized_caldidates_array[x] = x;
         int h_x_p = histogram_array[2*x];
         int h_x_n = histogram_array[2*x-1];
         if (  h_x_p == 0 || h_x_n == 0 ) {
            scores_array[x] = max(h_x_p, h_x_n);
         } else {
            scores_array[x] = h_x_p * h_x_n;
         }
      }
   } else { //* CPU sequential implementation of assign_scores
      cout << "ERROR: invalid mode in aiassign_scores()\n";
      exit(0);
   }
  
   #ifdef DEBUG
      cout << "DEBUGGING: run_assign_scores_kernel finished.\n" << endl;
      print_authorized_caldidates_array();
      print_scores_array();
   #endif
      
}

bool compareScores(pair<int, int> i1, pair<int, int> i2) {
    return (i1.second < i2.second);
}

void Preprocessor::sort_authorized_caldidates_according_to_scores() {
   
   if (mode==0) { //* CPU sequential sorting of authorized_caldidates
      pair<int, char> pairt[num_vars];
      for (int i = 1; i < num_vars + 1; i++) {
         pairt[i-1].first = authorized_caldidates_array[i];
         pairt[i-1].second = scores_array[i];
      }
      // Sorting the pair array.
      sort(pairt, pairt + num_vars, compareScores);
      // Modifying original arrays
      for (int i = 1; i < num_vars + 1; i++) {
         authorized_caldidates_array[i] = pairt[i-1].first;
         scores_array[i] = pairt[i-1].second;
      }
   } else if (mode==1) { //* GPU parallel sorting of authorized_caldidates
      run_sort_wrt_scores_kernel();
   } else {
      cout << "ERROR: sort_authorized_caldidates_according_to_scores()\n";
      exit(0);
   }

   #ifdef DEBUG
      cout << "DEBUGGING: sort_authorized_caldidates_according_to_scores finished.\n" << endl;
      print_authorized_caldidates_array();
      print_scores_array();
   #endif

}

void Preprocessor::prune() {
   //* No need to parallleize as O(logn) TODO: rethink this decision
   int x = mu;
   long * elem = upper_bound(scores_array + 1, scores_array + num_vars + 1, x);
   cutoffpoint = (((long)(elem - scores_array)));
   
   #ifdef DEBUG
      cout << "* elem: " << *elem << "\n";
      cout << "DEBUGGING: prune finished.\n" << endl;
      cout << "DEBUGGING: cutoffpoint: " << cutoffpoint << "\n" << endl;
   #endif

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
        
        
        
         int size = occur_list_p.getOccurListSize();
         for (int i=0 ;i<size; i++){
            int clause_index = occur_list_p.getClauseIndex(i);
            Clause& c = cnf->getClause(clause_index);     
            int nlits = c.getNumLits();
            for (int j=0; j<nlits; j++) {
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
         
         

         size = occur_list_n.getOccurListSize();
         for (int i=0 ;i<size; i++) {
            int clause_index = occur_list_n.getClauseIndex(i);
            Clause& c = cnf->getClause(clause_index);     
            int nlits = c.getNumLits();
            for (int j=0; j<nlits; j++) {
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

   #ifdef DEBUG
      cout << "DEBUGGING: LCVE_algorithm finished  \n\n";
      print_elected_candidates_vector();
   #endif

}


//* TODO: parallelize the cureation of occur table
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
   
   #ifdef DEBUG
      cout << "DEBUGGING: OccurTable created  \n\n";
      print_occur_table();
   #endif

}

// * Algorithm 4
void Preprocessor::BVIPE_algorithm() {
   
   eliminated_array = new bool[getNumVars()];
   for (int i=0; i<getNumVars(); i++) {
      eliminated_array[i] = false;
   }
   int numTautologies = 0;
   int numResolvents = 0;
   int numDeleted = 0;

   // if (mode==0) { //* CPU sequential BVIPE_algorithm
      
      
      for(auto var :elected_candidates_vector) {
         eliminated_array[var] = false;
         numTautologies = 0;
         int index_p = 2*var;
         int index_n = 2*var-1; 
         if (histogram_array[index_p] == 1 || histogram_array[index_n] == 1) {
            Resolve(var);
            eliminated_array[var] = true;
         }
         else {
            numTautologies = TautologyLookAhead(var);
            numResolvents = histogram_array[index_p] * histogram_array[index_n];
            numDeleted = histogram_array[index_p] + histogram_array[index_n];
            if(numResolvents - numTautologies < numDeleted){
               Resolve(var);
               eliminated_array[var] = true;
            }
         }
        
         #ifdef DEBUG
            cout << "var: " << var << endl;
            cout << " histogram_array[index_p] :" <<  histogram_array[index_p] << endl;
            cout << " histogram_array[index_n] :" <<  histogram_array[index_n] << endl;
            cout << "numTautologies : " << numTautologies << endl;
            cout << "numResolvents : " << numResolvents << endl;
            cout << "numDeleted : " << numDeleted << endl;
         #endif

      }
   // } 
   
   // else if (mode==1) { //* GPU parallel BVIPE_algorithm
   //    //* TODO: complete
   // } else {
   //    cout << "ERROR: invalid mode in BVIPE_algorithm\n";
   //    exit(0);
   // }

   #ifdef DEBUG
      cout << "DEBUGGING: BVIPE_algorithm finished.\n" << endl;
      print_eliminated_array();
   #endif

}

void Preprocessor::Resolve(int x){
   int index_p = 2*x;
   int index_n = 2*x-1;
   OccurList& occur_list_p = occur_table->getOccurList(index_p);
   OccurList& occur_list_n = occur_table->getOccurList(index_n);
   // vector< Clause *> resolvents;
   for(int i=0; i< occur_list_p.getOccurListSize(); i++ ){
      for(int j=0; j< occur_list_n.getOccurListSize(); j++ ){
         int c1_ind = occur_list_p.getClauseIndex(i);
         int c2_ind = occur_list_n.getClauseIndex(j);
         Clause& c1 = cnf->getClause(c1_ind);
         Clause& c2 = cnf->getClause(c2_ind);
         set < int > newClause;
         for(int k=0; k<c1.getNumLits(); k++){
            if(c1.getLit(k) != index_p && c1.getLit(k) != index_n)
               newClause.insert(c1.getLit(k));
         }
         for(int k=0; k<c2.getNumLits(); k++){
            if(c2.getLit(k) != index_p && c2.getLit(k) != index_n)
               newClause.insert(c2.getLit(k));
         }
         if(!IsTautology(newClause)){
            Clause* c = new Clause();
            c->setNumLits(newClause.size());
            c->initializeArray();
            int k=0;
            for(auto lit: newClause){
               c->setLit(k, lit);
               k++;
            }
            resolvents.push_back(c);
         }
         c1.setDeletedFlag(true);
         c2.setDeletedFlag(true);
      }
   }
}

bool Preprocessor::IsTautology(set<int> Clause){
   int prev = 0;
   for(auto it=Clause.begin(); it!=Clause.end(); it++){
      if(prev!=0 && prev%2==1 && prev + 1 ==*it){
         return true;
      }
      prev = *it;
   }
   return false;
}

int Preprocessor::TautologyLookAhead(int x){
   int index_p = 2*x;
   int index_n = 2*x - 1;
   int numTautologies = 0;
   OccurList& occur_list_p = occur_table->getOccurList(index_p);
   OccurList& occur_list_n = occur_table->getOccurList(index_n);
   // vector< Clause *> resolvents;
   for(int i=0; i< occur_list_p.getOccurListSize(); i++ ){
      for(int j=0; j< occur_list_n.getOccurListSize(); j++ ){
         int c1_ind = occur_list_p.getClauseIndex(i);
         int c2_ind = occur_list_n.getClauseIndex(j);
         Clause& c1 = cnf->getClause(c1_ind);
         Clause& c2 = cnf->getClause(c2_ind);
         set < int > newClause;
         for(int k=0; k<c1.getNumLits(); k++){
            if(c1.getLit(k) != index_p && c1.getLit(k) != index_n)
               newClause.insert(c1.getLit(k));
         }
         for(int k=0; k<c2.getNumLits(); k++){
            if(c2.getLit(k) != index_p && c2.getLit(k) != index_n)
               newClause.insert(c2.getLit(k));
         }
         // c1.print_clause();
         // c2.print_clause();
         // std::cout << "New Clause: ";
         for(auto x: newClause){
            // std::cout << x << " ";
         }
         // std::cout << "\n";
         if(IsTautology(newClause)){
            numTautologies++;
         }
      }
   }
   return numTautologies;
}


//***--------------------------dimacs-file-print-function------------------------------------------------***

void Preprocessor::printCnfInDimacsFile() {
   char cwd1[1000];
   getcwd(cwd1, sizeof(cwd1));
   char * dimacs_out_file_name = "/removed_vars.cnf\0";
   char * path_to_executable = strcat(cwd1, dimacs_out_file_name);
   FILE * fptr;
   fptr = fopen(path_to_executable, "w");
   // int num_clauses;
   // int num_vars;
   int l_num_clauses=0;
   vector<Clause*> remaining_clauses;
   for (int i=0; i<num_clauses;i++) {
      Clause & c = cnf->getClause(i);
      if (!c.getDeletedFlag()) {
         l_num_clauses++;
         remaining_clauses.push_back(&c);
      }
   }
   l_num_clauses += resolvents.size();
   int l_num_vars = num_vars;
   int var = num_vars;
   for (int i=0; i<num_vars; i++){
      if (eliminated_array[i]){
         l_num_vars--;
      
         for (auto c : remaining_clauses) {
            int c_num_lits = c->getNumLits();
            for (int j=0; j<c_num_lits; j++) {
               if (c->getLit(j)==2*var) {
                  c->setLit(j, 2*i);
               } else if (c->getLit(j)==2*var-1) {
                  c->setLit(j, 2*i-1);
               }
            }
         }
        
         for (auto c : resolvents) {
            int c_num_lits = c->getNumLits();
            for (int j=0; j<c_num_lits; j++) {
               if (c->getLit(j)==2*var) {
                  c->setLit(j, 2*i);
               } else if (c->getLit(j)==2*var-1) {
                  c->setLit(j, 2*i-1);
               }
            }
         }

         var--;
      }
      
   }

   fprintf(fptr,"p cnf %d %d\n", l_num_vars, l_num_clauses);
   for (auto c : remaining_clauses) {
      int c_num_lits = c->getNumLits();
      for (int j=0; j<c_num_lits; j++) {
         int lit = c->getLit(j); 
         if (lit%2 == 0) {  
            fprintf(fptr,"%d ", lit/2);
         }
         else {  
            fprintf(fptr,"%d ", -((lit/2) + 1));
         }
      }
      fprintf(fptr,"0\n");
   }

   for (auto c : resolvents) {
      int c_num_lits = c->getNumLits();
      for (int j=0; j<c_num_lits; j++) {
         int lit = c->getLit(j); 
         if (lit%2 == 0) {  
            fprintf(fptr,"%d ", lit/2);
         }
         else {  
            fprintf(fptr,"%d ", -((lit/2) + 1));
         }
      }
      fprintf(fptr,"0\n");
   }

   // fprintf(fptr,"Non Termination due to inline monitoring in mutant_id: %d , Reachable State: tmpvalue = %d, exp = %d\n", mutant_id, *tmpvalue, *exp);

}



//***--------------------------debug-print-functions----------------------------------------------------***

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

void Preprocessor::print_eliminated_array() {
   cout << "DEBUGGING: eliminated_array: \n";
   cout << "getNumVars(): " << getNumVars() << endl;
   for (int i=0;i<getNumVars();i++){
      cout << eliminated_array[i] << ", \n";
   }
   cout << endl;
}