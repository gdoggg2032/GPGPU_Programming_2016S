all:
	nvcc -std=c++11 -arch=sm_30 -O2 -c lab2.cu -o lab2.o -I../utils
	nvcc -std=c++11 -arch=sm_30 -O2 main.cu lab2.o -I../utils
run:
	rm result.mkv
	./a.out
	avconv -i result.y4m result.mkv
convert:
	avconv -i result.y4m result.mkv

