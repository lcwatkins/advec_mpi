
CC=mpiCC
CFLAGS=-fopenmp -std=c++11

advect2d.o: advect2d.cpp 
	$(CC) $(CFLAGS) advect2d.cpp -o advect2d.o

