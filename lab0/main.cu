#include <cstdio>
#include <cstdlib>
#include "SyncedMemory.h"

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}


__global__ void SomeTransform(char *input_gpu, int fsize) {

	// this function will transform the first 64 characters to the next alphabet.
    // e.g., A to B, Z to A, a to b and z to a.
    // only transform alphabet, punctuation and space are excluded.
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < fsize and input_gpu[idx] != '\n' and input_gpu[idx] != ' ') {
		if (input_gpu[idx] >= 65 and input_gpu[idx] < 91) {
			// if uppercase
			input_gpu[idx] = (input_gpu[idx] - 65 + 1) % 26 + 65;
		} else if (input_gpu[idx] >= 97 and input_gpu[idx] < 123){
			// if lowercase
			input_gpu[idx] = (input_gpu[idx] - 97 + 1) % 26 + 97;

		}
	}
}

int main(int argc, char **argv) {
	// init, and check
	if (argc != 2) {
		printf("Usage %s <input text file>\n", argv[0]);
		abort();
	}
	FILE *fp = fopen(argv[1], "r");
	if (not fp) {
		printf("Cannot open %s", argv[1]);
		abort();
	}
	// get file size
	fseek(fp, 0, SEEK_END);
	size_t fsize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	// read files
	MemoryBuffer<char> text(fsize+1);
	auto text_smem = text.CreateSync(fsize);
	CHECK;
	fread(text_smem.get_cpu_wo(), 1, fsize, fp);
	text_smem.get_cpu_wo()[fsize] = '\0';
	fclose(fp);

	// TODO: do your transform here
	char *input_gpu = text_smem.get_gpu_rw();
	// An example: transform the first 64 characters to '!'
	// Don't transform over the tail
	// And don't transform the line breaks
	SomeTransform<<<2, 32>>>(input_gpu, fsize);

	puts(text_smem.get_cpu_ro());
	return 0;
}
