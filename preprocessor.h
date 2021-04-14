#include <iostream>
#include <fstream>
#include <ctime>
#include <string.h>
#include <set>
#include <sstream>

using namespace std;

//* main data structures file




typedef int Var;
typedef int Lit;
#define Neg(l) (l & 1)

//* clause
class Clause {

   //* pointer to array of literals of this clause
   //* TODO: make literal_array private
   Lit * literal_array;
   int num_lits; 

public:  
   
   Clause() {};
   
   void print_clause();
   // {
   //    for(int i=0; i<num_lits; i++){
   //       cout << literal_array[i] << " ";
   //    }
   //    cout << "\n";
   // }
   void setNumLits(int n) {num_lits = n;}
   int getNumLits() {return num_lits;}
   int getLit(int index) { return literal_array[index]; }
   void setLit(int index, int val) { literal_array[index] = val; }
   void initializeArray() { literal_array = new int[num_lits]; }
};

// class OccurList {
   
//    //*
//    int * clause_index_array;
//    int occur_list_size;

// public:
   
//    // get_occur_list_size() {

//    // }
//    // set_occur_list_size() {

//    // }

// };

// /* OccurTab: provides interface to efficiently iterate over all clauses containing a particular literal. 
//  * occurTAB instances can be constructed in parallel.
//  */
// class OccurTab {
   
//    //* pointer to array of occurList
//    OccurList * occur_list_array;

// public:

// };

//* stores the input CNF formula
class Cnf {
   //* TODO: make clause_array private
   //* pointer to the array of clauses of this CNF formula
   Clause * clause_array;

   int num_clauses;
   int num_vars;
   int num_lits;

public:
   Cnf()  {};
   void print_cnf(){
      for(int i=0; i<num_clauses; i++){
         cout << "Clause no : " << i << " \nClause: ";
         clause_array[i].print_clause();
      }
   }
   

   void setNumClauses(int s){ num_clauses = s;}
   int getNumClauses(){ return num_clauses; }
   void setNumVars(int s){ num_vars = s;}
   int getNumVars(){ return num_vars; }
   void setNumLits(int n) {num_lits = n;}
   int getNumLits() {return num_lits;}
   Clause& getClause(int index) { return clause_array[index]; }
   void setClause(int index, Clause& val) { clause_array[index] = val; }
   void initializeArray() { clause_array = new Clause[num_clauses]; }
};


//* the main preprecessor class
class Preprocessor {

   //* input
   Cnf * cnf; // Not null after constructor.
   int num_vars;

   //* hyper-parameter TODO: decide mu's valuse
   int mu;

 

   
public:
   Preprocessor() {};
   //* TODO: make these algo related data structures as private
   //* Algorithm related data structures
   //* TODO: think: int * v/s Var * for authorized_caldidates_array
   int * authorized_caldidates_array; // Not after constructor.
   int * histogram_array; // Not after constructor.
   int * scores_array; // Not after constructor.
  
  
   //* TODO: put algo related functions in private and make a wrapper in public
   
   //* Algorithm 1 functions
   void constructAGPU();
   void initialize_hisotgram_array();
   void assign_scores();
   void sort_authorized_caldidates_according_to_scores();
   void prune();


   void setNumVars(int s){ num_vars = s;}
   int getNumVars(){ return num_vars; }
   // void read_cnf(ifstream& in);
   // void initialize();
   size_t cnf_size() { return cnf->getNumClauses(); }
   void initializeCnf() { 	cnf = new Cnf(); }
   Cnf * getCnf() { return cnf; }
   void print_cnf(){
      cnf->print_cnf();
   }
   void add_clause(Clause* c, int index);
};

// Preprocessor P;

inline unsigned int v2l(int i) { // maps a literal as it appears in the cnf to literal
	if (i < 0) return ((-i) << 1) - 1; 
	else return i << 1;
} 

inline Var l2v(Lit l) {
	return (l+1) / 2;	
} 

inline Lit negate_(Lit l) {
	if (Neg(l)) return l + 1;  // odd
	return l - 1;		
}

inline int l2rl(int l) {
	return Neg(l)? -((l + 1) / 2) : l / 2;
}

inline void Abort(string s, int i) {
	cout << endl << "Abort: ";
	switch (i) {
	case 1: cout << "(input error)" << endl; break;
	case 2: cout << "command line arguments error" << endl; break;
	case 3: break;
	default: cout << "(exit code " << i << ")" << endl; break;
	}
	cout << s << endl;
	exit(i);
}

inline bool match(ifstream& in, char* str) {
    for (; *str != '\0'; ++str)
        if (*str != in.get())
            return false;
    return true;
}

void read_cnf(ifstream& in);
void initialize();
void add_clause(Clause* c, int index);