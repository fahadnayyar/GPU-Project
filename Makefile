default: preprocessor

preprocessor:  preprocessor.cpp preprocessor_kernels.cu preprocessor.h preprocessor_kernels.h main.cpp
	nvcc preprocessor.cpp preprocessor_kernels.cu main.cpp  -o preprocessor

run:
	./preprocessor

clean:
	rm preprocessor