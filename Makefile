#  @author: Krishna Kariya, Fahad Nayyar, 2021

default: preprocessor

preprocessor:  preprocessor.cpp preprocessor_kernels.cu preprocessor.h main.cpp
	nvcc -g preprocessor.cpp preprocessor_kernels.cu main.cpp  -o preprocessor

run:
	./preprocessor

clean:
	rm preprocessor