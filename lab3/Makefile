all:
	nvcc -std=c++11 -arch=sm_30 -O2 -c lab3.cu -o lab3.o -I../utils
	nvcc -std=c++11 -arch=sm_30 -O2 -c pgm.cpp -o pgm.o -I../utils
	nvcc -std=c++11 -arch=sm_30 -O2 main.cu lab3.o pgm.o -I../utils -D_MWAITXINTRIN_H_INCLUDED
run:
	./a.out img_background.ppm img_target.ppm img_mask.pgm 130 600 output.ppm
	convert output.ppm result/sample.png

