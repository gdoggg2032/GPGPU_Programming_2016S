all:
	nvcc -std=c++11 -ccbin g++-4.9 -arch=sm_30 -O2 -c counting.cu -o counting.o -I../utils 
	nvcc -std=c++11 -ccbin g++-4.9 -arch=sm_30 -O2 main.cu counting.o -I../utils
main:
	nvcc -std=c++11 -arch=sm_30 -O2 main.cu counting.o -I../utils
run:
	./a.out
t:
	nvcc -std=c++11 -arch=sm_30 -O2 test.cu -I../utils -o ./test
	./test
