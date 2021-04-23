//* TODO: free all data structures before exit !!
//* TODO: add timing code in the implementation itself. Also calculate per kernel time

#include "preprocessor.h"
#include <ctime>
#include <cuda_runtime.h>

using namespace std;

#ifdef _MSC_VER
#include <ctime>
static inline double cpuTime(void) {
    return (double)clock() / CLOCKS_PER_SEC; }
#else
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
static inline double cpuTime(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1000000; }
#endif

//* global variables
double begin_time;
Preprocessor P;

int main ( int argc, char **argv	) {
   char * mode = getenv ( "MODE" );
	if(mode == NULL){
		P.setMode(0);
	}
	else{
		int mode_int = atoi( mode );
		P.setMode(mode_int);
	}
	if (P.getMode()==0){
		cout << "running the sequential CPU implementation of variable elimination\n";
	}else if (P.getMode()==1){
		cout << "running the parallel GPU implementation of variable elimination\n";
	}else {
		exit(0);
		cout << "ERROR: invalid mode\n";
	}
   begin_time = cpuTime();
	ifstream in (argv[argc - 1]);
	if (!in.good()) Abort("cannot read input file", 1);	
	read_cnf(in);		
	in.close();
	#ifdef DEBUG
		P.print_cnf();
	#endif
   


	if (P.getMode()==0){
		#ifdef ENABLE_TIMER
			struct timespec start_cpu, end_cpu;
			float msecs_cpu;
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_cpu);
		#endif
				P.do_parallel_preprocessing();
		#ifdef ENABLE_TIMER
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_cpu);
			msecs_cpu = 1000.0 * (end_cpu.tv_sec - start_cpu.tv_sec) + (end_cpu.tv_nsec - start_cpu.tv_nsec)/1000000.0;
			cout<<"CPU sequential sat simplification done in "<<msecs_cpu<<" milliseconds.\n"<<flush;
		#else
			cout<<"CPU sequential sat simplification done.\n"<<flush;
		#endif
	}else if (P.getMode()==1){
		#ifdef ENABLE_TIMER
			cudaEvent_t start_gpu, end_gpu;
			float msecs_gpu;
			cudaEventCreate(&start_gpu);
			cudaEventCreate(&end_gpu);
			cudaEventRecord(start_gpu, 0);
		#endif
				P.do_parallel_preprocessing();
		#ifdef ENABLE_TIMER
			cudaEventRecord(end_gpu, 0);
			cudaEventSynchronize(end_gpu);
			cudaEventElapsedTime(&msecs_gpu, start_gpu, end_gpu);
			cudaEventDestroy(start_gpu);
			cudaEventDestroy(end_gpu);
			cout<<"GPU parallel sat simplification done in "<<msecs_gpu<<" milliseconds.\n";
		#else
			cout<<"GPU parallel sat simplification done.\n"<<flush;
		#endif
	}
	
	return 0;
}

//****----------------------------------CNF-parsing-code---------------------------------**** 

void skipLine(ifstream& in) {
	for (;;) {
		//if (in.get() == EOF || in.get() == '\0') return;
		if (in.get() == '\n') { return; }
	}
}

static void skipWhitespace(ifstream& in, char&c) {
	c = in.get();
	while ((c >= 9 && c <= 13) || c == 32)
		c = in.get();
}

static int parseInt(ifstream& in, char &c) {
	int     val = 0;
	bool    neg = false;
	if (c == '-') neg = true, c = in.get();
	if (c < '0' || c > '9') cout << c, Abort("Unexpected char in input", 1);
	while (c >= '0' && c <= '9')
		val = val * 10 + (c - '0'),
		c = in.get();
	return neg ? -val : val;
}

void read_cnf(ifstream& in) {
	int i;
	unsigned int vars, clauses, unary = 0;
	set<Lit> s;
	Clause* c;
	char d;
	bool flag = false;
	int index=0;

	while (in.peek() == 'c') skipLine(in);

	if (!match(in, "p cnf")) Abort("Expecting `p cnf' in the beginning of the input file", 1);
	in >> vars; // since vars is int, it reads int from the stream.
	in >> clauses;
	if (!vars || !clauses) Abort("Expecting non-zero variables and clauses", 1);
	cout << "vars: " << vars << " clauses: " << clauses << endl;
	P.initializeCnf();
	P.getCnf()->setNumClauses(clauses);
	P.getCnf()->setNumVars(vars);
	P.setNumVars(vars);
	P.setNumClauses(clauses);
	initialize();

	while (in.good() && in.peek() != EOF) {
		skipWhitespace(in, d);
		if(in.peek() == EOF) {flag=true;};		
		if(flag == false) {
			i = parseInt(in, d);
		}else {
			if(d == '0') {
				i = 0;
			 }else{
			 	break;
			 }
		}
		if (i == 0) {
			c = new Clause();
			c->setNumLits(s.size());
            c->initializeArray();
			int j=0;
			for(auto lit: s)
			{
				c->setLit(j, lit);
				j++;				
			}
			switch (s.size()) {
				case 0: {
					stringstream num;
					num << P.cnf_size() + 1;
					Abort("Empty clause not allowed in input formula (clause " + num.str() + ")", 1);
				}
				default: {
					add_clause(c, index);
					index++;
				}
			}
			c = NULL;
			s.clear();
			continue;
		}
		i = v2l(i);		
		s.insert(i);
		if(flag == true){
			break;
		}
	}	
	cout << "Read " << P.cnf_size() << " clauses in " << cpuTime() - begin_time << " secs." << endl << "Solving..." << endl;
}


void initialize() {	
	P.getCnf()->setNumLits(2 * P.getCnf()->getNumLits());
	P.getCnf()->initializeArray();
}

void add_clause(Clause* c, int index) {	
	P.getCnf()->setClause(index, *c);
}


