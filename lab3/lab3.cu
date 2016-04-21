#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
		const float *background,
		const float *target,
		const float *mask,
		float *output,
		const int wb, const int hb, const int wt, const int ht,
		const int oy, const int ox
		)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (0 <= yt && 0 <= xt && yt < ht && xt < wt && mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void PoissonImageCloningIteration(
		float *fixed, 
		const float *mask,
		float *buf1, float *buf2, // buf1 -> buf2
		const int wt, const int ht
		)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (0 <= yt && 0 <= xt && yt < ht && xt < wt) { 
		if(mask[curt] > 127.0f) {
			int curtN = wt*(yt-1)+xt;
			int curtW = wt*yt+(xt-1);
			int curtS = wt*(yt+1)+xt;
			int curtE = wt*yt+(xt+1);

			for(int c = 0; c < 3; c++) {

				float sumCurrent = 0.0f;
				if(yt > 0 && mask[curtN] > 127.0f) sumCurrent += buf1[3*curtN+c];
				if(xt > 0 && mask[curtW] > 127.0f) sumCurrent += buf1[3*curtW+c];
				if(yt < ht-1 && mask[curtS] > 127.0f) sumCurrent += buf1[3*curtS+c];
				if(xt < wt-1 && mask[curtE] > 127.0f) sumCurrent += buf1[3*curtE+c];


				float pixel_around = 4.0f;
		//		if(yt == 0 || yt == ht-1) pixel_around -= 1.0;
		//		if(xt == 0 || xt == wt-1) pixel_around -= 1.0;
				float cb_next = (sumCurrent + fixed[curt*3+c]) / pixel_around;
				buf2[3*curt+c] = cb_next;//min(255.0, max(0.0, cb_next));
			}
		}
		/*else {
			for(int c = 0; c < 3; c++) {
				buf2[3*curt+c] = fixed[3*curt+c];
			}
		}*/



	}
}

__global__ void CalculateFixed(
		const float *background,
		const float *target,
		const float *mask,
		float *fixed,
		const int wb, const int hb, const int wt, const int ht,
		const int oy, const int ox
		)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	//if (yt < ht and xt < wt and mask[curt] < 127.0f) {
	if (yt >= 0 && xt >= 0 && yt < ht && xt < wt) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if(0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			if(mask[curt] < 127.0f) {
				// set fixed to background
				for(int c = 0; c < 3; c++) {
					fixed[3*curt+c] = background[3*curb+c];
				}
			}
			else {
				int curtN = wt*(yt-1)+xt;
				int curtW = wt*yt+(xt-1);
				int curtS = wt*(yt+1)+xt;
				int curtE = wt*yt+(xt+1);

				int curbN = wb*(yb-1)+xb;
				int curbW = wb*yb+(xb-1);
				int curbS = wb*(yb+1)+xb;
				int curbE = wb*yb+(xb+1);

				for(int c = 0; c < 3; c++) {
					// count target gradient
					float tgradient = 0.0f;
					if(yt > 0) tgradient += target[3*curt+c] - target[3*curtN+c];
					if(xt > 0) tgradient += target[3*curt+c] - target[3*curtW+c];
					if(yt < ht-1) tgradient += target[3*curt+c] - target[3*curtS+c];
					if(xt < wt-1) tgradient += target[3*curt+c] - target[3*curtE+c];

					// count border fixed value
					float vborder = 0.0f;
					if(yt == 0 || mask[curtN] < 127.0f) vborder += background[3*curbN+c];
					if(xt == 0 || mask[curtW] < 127.0f) vborder += background[3*curbW+c];
					if(yt == ht-1 || mask[curtS] < 127.0f) vborder += background[3*curbS+c];
					if(xt == wt-1 || mask[curtE] < 127.0f) vborder += background[3*curbE+c];

					fixed[3*curt+c] = tgradient + vborder;

				}
			}
		}

	}
}
void PoissonImageCloning(
		const float *background,
		const float *target,
		const float *mask,
		float *output,
		const int wb, const int hb, const int wt, const int ht,
		const int oy, const int ox
		)
{
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);

	CalculateFixed<<<gdim, bdim>>>(
			background, target, mask, fixed,
			wb, hb, wt, ht, oy, ox
			);
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

	for(int i = 0; i < 6000; i++) {
		PoissonImageCloningIteration<<<gdim, bdim>>>(
				fixed, mask, buf1, buf2, wt, ht
				);
		PoissonImageCloningIteration<<<gdim, bdim>>>(
				fixed, mask, buf2, buf1, wt, ht
				);
	}

	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);

	SimpleClone<<<gdim, bdim>>>(
			background, buf1, mask, output,
			wb, hb, wt, ht, oy, ox
			);

	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);

}

