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
#include <thrust/unique.h>

#define D 10

struct is_one
{
	__host__ __device__
		bool operator()(const int x)
		{
			return (x == 1);
		}
};

struct is_continuous
{
        __host__ __device__
                bool operator()(const int a, const int b)
                {
                        return (a+1 == b) && (a != 0) || (a==0 && b==0);
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

__global__ void headTagging(int *head, int n_head, int* output, int text_size)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n_head)return;
        if(idx == 0)return;
        output[head[idx]] = 1;
}

__global__ void reverseEachString(char *text, int text_size, int* head_tail, int n_string, int* head_label)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= text_size)return;

        // get bound
        int head_idx = head_label[idx];
        int left = head_tail[2 * head_idx];
        int right = head_tail[2 * head_idx + 1];

        //do swap corresponding characters
        //get index in this string
        int idx_in_string = idx-left;
        int string_length = right - left;
        //if idx_in_string exceeds this string's length, return
        if(idx_in_string >= string_length)
                return;

        int idx_another = string_length - 1 - idx_in_string;

        char tmp = text[left + idx_in_string];
        text[left + idx_in_string] = text[left + idx_another];
        text[left + idx_another] = tmp;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
	thrust::device_ptr<char> text_d(text);
	thrust::device_ptr<int> pos_d(pos);
	thrust::device_ptr<int> head_d(head);
	
	thrust::device_vector<int> index_d(text_size);
        thrust::sequence(index_d.begin(), index_d.end());

        thrust::pair<thrust::device_ptr<int>, typeof(index_d.end())> new_end;
        new_end = thrust::unique_by_key(pos_d, pos_d + text_size, index_d.begin(), is_continuous());
        if((new_end.second - index_d.begin()) % 2 != 0)
        {
                index_d[new_end.second - index_d.begin()] = text_size;
                new_end.second += 1;
        }

	thrust::device_vector<int> head_label(text_size, 0);

        headTagging<<<n_head/1024 + 1, 1024>>>(head, n_head, thrust::raw_pointer_cast(head_label.data()), text_size);
        cudaDeviceSynchronize();
        thrust::device_vector<int> class_map(text_size, 0);
        thrust::inclusive_scan(head_label.begin(), head_label.end(), head_label.begin());


        reverseEachString<<<39063, 1024>>>(text, text_size, thrust::raw_pointer_cast(index_d.data()), n_head, thrust::raw_pointer_cast(head_label.data()));
        cudaDeviceSynchronize();
		
}
