#include <stdio.h>
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

#define D 10

struct is_one
{
	__host__ __device__
		bool operator()(const int x)
		{
			return (x == 1);
		}
};

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__device__ bool BIT[40000500][D];

__global__ void BITBuilding_depth(const char *text, int text_size, int d)
{
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);
	if(idx >= text_size)
		return;
	if(d == 0)
	{
		if(text[idx] != '\n')
			BIT[idx][d] = 1;
		else
			BIT[idx][d] = 0;
	}
	else
	{
		int dim = (int)pow(2, d);
		if(idx < text_size/dim)
			BIT[idx][d] = (BIT[2*idx][d-1] && BIT[(2*idx+1)][d-1]);
	}
}

__global__ void Counting(const char *text, int *pos, int text_size)
{
	int idx = (blockIdx.x * blockDim.x + threadIdx.x) % text_size;
	if(idx >= text_size)
		return;
	if(BIT[idx][0] == 0)
		pos[idx] = 0;//*(pos+idx) = 0;
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
			if(BIT[(index-1)/2][ d+1] == 1)
			{
				add *= 2;
				d += 1;
				index = (index-1)/2;
			}
			else
				break;
		}

		while(index >= 0 && add > 0 && d >= 0)
		{
			if(BIT[index][ d] == 1)
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
		pos[idx] = len;
	}

}

void CountPosition(const char *text, int *pos, int text_size)
{
	for(int d = 0; d < D; d++)
	{
		BITBuilding_depth<<<39063, 1024>>>(text, text_size, d);
		cudaDeviceSynchronize();
	}
	Counting<<<39063, 1024>>>(text, pos, text_size );
}

int ExtractHead(const int *pos, int *head, int text_size)
{
	int nhead;
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head);

	thrust::device_vector<int> index_d(text_size);
	thrust::sequence(index_d.begin(), index_d.end());

	thrust::device_ptr<int> ret = thrust::copy_if(index_d.begin(), index_d.end(), pos_d, head_d, is_one());

	nhead = ret - head_d;
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
}
