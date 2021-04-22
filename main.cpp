//* TODO: free all data structures before exit !!

#include "preprocessor.h"


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
   //* parsing code copied from EDUSAT (our base SAT solver)
	char * mode = getenv ( "MODE" );
	if(mode == NULL){
		P.setMode(0);
	}
	else{
		int mode_int = atoi( mode );
		P.setMode(mode_int);
	}
   begin_time = cpuTime();
	ifstream in (argv[argc - 1]);
	if (!in.good()) Abort("cannot read input file", 1);	
	read_cnf(in);		
	in.close();
	P.print_cnf();
   P.do_parallel_preprocessing();
	return 0;
}

//* parsing code 

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
	// char c;
	// skipWhitespace(in, c);
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
	// cnf = new Cnf();
	P.initializeCnf();
	P.getCnf()->setNumClauses(clauses);
	P.getCnf()->setNumVars(vars);
	P.setNumVars(vars);
	P.setNumClauses(clauses);
	// set_nvars(vars);
	// set_nclauses(clauses);
	initialize();

	while (in.good() && in.peek() != EOF) {
		// i = parseInt(in);
		skipWhitespace(in, d);
		
		if(in.peek() == EOF) {flag=true;};		
		
		if(flag == false)
		{
			i = parseInt(in, d);
		}else
		{ //cout << d << endl; 
			// i = (int)d;
			if(d == '0') {
				// Abort("Clause Line did not ended with zero. ", 1);
			 	// break; 
			 	i = 0;
			 }else{
			 	break;
			 }
			// i = 0; 
		}
		if (i == 0) {
			c = new Clause();
			c->setNumLits(s.size());
            c->initializeArray();
			// c->literal_array = new Lit[s.size()];
			// cout << "Size of clause: " << s.size() << " ";
			int j=0;
			for(auto lit: s)
			{
				//* TODO: doubt fix this s.find()
				c->setLit(j, lit);  // [j] = lit;
				j++;				
			}
			// c.cl().resize(s.size());
			// copy(s.begin(), s.end(), c.cl().begin());
			switch (s.size()) {
			case 0: {
				stringstream num;  // this allows to convert int to string
				num << P.cnf_size() + 1; // converting int to string.
				Abort("Empty clause not allowed in input formula (clause " + num.str() + ")", 1); // concatenating strings
			}
			// case 1: {
			// 	Lit l = c.cl()[0];
			// 	// checking if we have conflicting unaries. Sufficiently rare to check it here rather than 
			// 	// add a check in BCP. 
			// 	if (state[l2v(l)] != VarState::V_UNASSIGNED)
			// 		if (Neg(l) != (state[l2v(l)] == VarState::V_FALSE)) {
			// 			S.print_stats();
			// 			Abort("UNSAT (conflicting unaries for var " + to_string(l2v(l)) +")", 0);
			// 		}
			// 	assert_lit(l);
			// 	add_unary_clause(l);
			// 	break; // unary clause. Note we do not add it as a clause. 
			// }
			default: {
				add_clause(c, index);
				index++;
			}
			}
			c = NULL;
			// c.reset();
			s.clear();
			continue;
		}
		// if (Abs(i) > vars) Abort("Literal index larger than declared on the first line", 1);
		// if (VarDecHeuristic == VAR_DEC_HEURISTIC::MINISAT) bumpVarScore(abs(i));
		i = v2l(i);		
		// if (ValDecHeuristic == VAL_DEC_HEURISTIC::LITSCORE) bumpLitScore(i);
		s.insert(i);
		if(flag == true){
			break;
		}
	}	
	// if (VarDecHeuristic == VAR_DEC_HEURISTIC::MINISAT) reset_iterators();
	cout << "Read " << P.cnf_size() << " clauses in " << cpuTime() - begin_time << " secs." << endl << "Solving..." << endl;
}


void initialize() {	
	P.getCnf()->setNumLits(2 * P.getCnf()->getNumLits());
	// cnf->clause_array = new Clause[cnf->getNumClauses()];
    P.getCnf()->initializeArray();
}

void add_clause(Clause* c, int index) {	
	// Assert(c.size() > 1) ;
	// c.lw_set(l);
	// c.rw_set(r);
	// int loc = static_cast<int>(cnf.size());  // the first is in location 0 in cnf	
	// int size = c.size();
	
	// watches[c.lit(l)].push_back(loc); 
	// watches[c.lit(r)].push_back(loc);
	// cnf.push_back(c);
	// cnf->clause_array[index] = *c;
    P.getCnf()->setClause(index, *c);
}


