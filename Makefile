all:
	nvcc -Xcompiler -fopenmp sparseMatrix.cu -o prog
clean:
	rm prog