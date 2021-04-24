# GPU-Project
Parallel simplification of SAT on GPUs

## Steps to build the preprocessor executable
make

## To run the preprocessor
$ MODE=1 ./preprocessor [path-to-dimacs-format-cnf-file]


NOTE1 : The new cnf file with removed variables is dumped in $pwd/removed_vars.cnf 

NOTE 2: Use MODE=0 for CPU (sequential) version. Use MODE=1 for GPU (parallle) version of preprocessor.

## To check correctness of preprocessor on one cnf input
Generate the removed_vars.cnf file by executing the preprocessor executable. Then execute the following command to invoke the MUVAL SAT-Solver on the removed_vars.cnf to check whether it is SAT or UNSAT.

## To check correctness of preprocessor on all the test CNF cases in this repository
$ python3 run_correctness_tests.py


NOTE: the output of speedup tests is stored in correctness_results.txt

## TO check parallel speedup
$ python3 run_speedup_tests.py


NOTE: the output of speedup tests is stored in speedup_results.txt

### The files authored by us are:
Makefile
preprocessor.h
preprocessor.cpp
preprocessor_kernels.cu
run_speedup_tests.py
run_correctness_tests.py
main.cpp
report.pdf
slides.pdf

