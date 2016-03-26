#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>

const int K = 500;
const int D = 9; // log(500)


__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__device__ int BIT[40000500][D];

__global__ void BITBuilding(const char *text, int *pos, int text_size)
{
	//printf("BITBuilding\n");	
	int idx = (blockIdx.x * blockDim.x + threadIdx.x) % text_size;
	//printf("building: index: %d\n", blockIdx.x * blockDim.x + threadIdx.x);
	//if(idx >= text_size)return;
	for(int d = 0; d < D; d++)
	{
		if(d == 0)
		{
			if(*(text+idx) != '\n')
				BIT[idx][d] = 1;
			else
				BIT[idx][d] = 0;

		}
		else
		{
			int dim = (int)pow(2, d);
			if(idx < text_size/dim)
			{
				BIT[idx][d] = (BIT[2*idx][d-1] && BIT[2*idx+1][d-1]);
			}

		}
		__syncthreads();
	}
}

__global__ void Counting(const char *text, int *pos, int text_size)
{
	//printf("Counting\n");
	int idx = (blockIdx.x * blockDim.x + threadIdx.x) % text_size;
	//printf("counting: index: %d\n", blockIdx.x * blockDim.x + threadIdx.x);
	//if(idx>=text_size)return;
	/* gpu part */
	if(BIT[idx][0] == 0)
		*(pos+idx) = 0;
	else
	{
		int len = 0;

		int d = 0;
		int add = 1;
		int index = idx;


		while(1)
		{


			if(index <= 0)
				break;
			if(index%2 == 0)
			{
				index -= 1;
				len += add;
			}

			if(BIT[(index-1)/2][d+1] == 1)
			{


				add *= 2;
				d += 1;
				index = (index-1)/2;
			}
			else
			{
				break;

			}

		}

		while(index >= 0 && add > 0 && d >= 0)
		{
			if(BIT[index][d] == 1)
			{
				// to left-down
				len += add;
				index = index * 2 - 1;
				d -= 1;

			}
			else
			{
				// to right-down
				index = index * 2 + 1;
				d -= 1;

			}

			add /= 2;
		}

		*(pos+idx) = len;
	}

//	__syncthreads();
}

void CountPosition(const char *text, int *pos, int text_size)
{
	int threadNum = text_size/2;
	BITBuilding<<<(text_size)/threadNum + 1, threadNum>>>(text, pos, text_size);
//	cudaDeviceSynchronize();
	Counting<<<(text_size)/threadNum + 1, threadNum>>>(text, pos, text_size);
}

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);
	//printf("0\n");
	// TODO
	//return 0;
	fprintf(stderr, "before TODO\n");
	//thrust::sequence(cumsum_d, cumsum_d + 3);
	thrust::device_vector<int> dev_vec(10, 1);
//	thrust::sequence(dev_vec.begin(), dev_vec.end());
//	size_t N = 3;
//	thrust::fill(cumsum_d, cumsum_d+N, (int) 1);
	fprintf(stderr, "TODO 0\n");
	//int *raw_ptr = thrust::raw_pointer_cast(cumsum_d);
	thrust::host_vector<int> host_vec(dev_vec.size());
	thrust::copy(dev_vec.begin(), dev_vec.end(), host_vec.begin());
	for(int i = 0; i < host_vec.size(); i++)
		fprintf(stderr, "vec[%d] == %d\n", i, host_vec[i]);
    //head[0] = 1;
	int test[100];
	cudaMemcpy(test, cumsum_d.get(), 100*sizeof(int), cudaMemcpyDeviceToHost);
//	thrust::copy_n(cumsum_d, text_size, head_d);
	fprintf(stderr, "TODO 1\n");
	for(int i = 0; i < 3; i++)
	{
		if(i == test[i])
			fprintf(stderr, "yes\n");
		else
			fprintf(stderr, "no: %d\n", test[i]);
	}
	fprintf(stderr, "TODO 2\n");
	
	
	*(head + 1) = 2;
	nhead = 2;
	fprintf(stderr, "after TODO\n");

	cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
}
