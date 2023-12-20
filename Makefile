all:
	nvcc -Xcompiler -fopenmp sparseMatrix.cu -o prog
	g++ traditionalSparseMatrix.cpp -o prog0
clean:
	rm prog prog0