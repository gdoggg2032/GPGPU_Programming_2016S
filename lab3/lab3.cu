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
		const float *target,
		float *strictInteriorPixels,
		float *bg,
		float *buf1, float *buf2, // buf1 -> buf2
		const int wt, const int ht
		)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (0 <= yt && 0 <= xt && yt < ht && xt < wt) { //and mask[curt] > 127.0f) {
		if(mask[curt] > 127.0f) {
			int curtN = wt*(yt-1)+xt;
			int curtW = wt*yt+(xt-1);
			int curtS = wt*(yt+1)+xt;
			int curtE = wt*yt+(xt+1);

			for(int c = 0; c < 3; c++) {
				
				float A = 0.0f;
				float B = 0.0f;
				if(yt > 0) {
					if(strictInteriorPixels[curtN] > 0.0f) A += buf1[3*curtN+c];
					else B += bg[curtN*3+c];
				}
				if(xt > 0) {
					if(strictInteriorPixels[curtW] > 0.0f) A += buf1[3*curtW+c];
					else B += bg[curtW*3+c];
				}
				if(yt < ht-1) {
					if(strictInteriorPixels[curtS] > 0.0f) A += buf1[3*curtS+c];
					else B += bg[curtS*3+c];
				}
				if(xt < wt-1) {
					if(strictInteriorPixels[curtE] > 0.0f) A += buf1[3*curtE+c];
					else B += bg[curtE*3+c];
				}

				/*float A = 0.0f;
				if(yt > 0 && mask[curtN] > 127.0f) A += buf1[3*curtN+c];
				if(xt > 0 && mask[curtW] > 127.0f) A += buf1[3*curtW+c];
				if(yt < ht-1 && mask[curtS] > 127.0f) A += buf1[3*curtS+c];
				if(xt < wt-1 && mask[curtE] > 127.0f) A += buf1[3*curtE+c];
				float B = 0.0f;
				if(yt > 0 && mask[curtN] < 127.0f) B += fixed[3*curtN+c];
				if(xt > 0 && mask[curtW] < 127.0f) B += fixed[3*curtW+c];
				if(yt < ht-1 && mask[curtS] < 127.0f) B += fixed[3*curtS+c];
				if(xt < wt-1 && mask[curtE] < 127.0f) B += fixed[3*curtE+c];*/
				float pixel_around = 4.0f;
				if(yt == 0 || yt == ht-1) pixel_around -= 1.0;
				if(xt == 0 || xt == wt-1) pixel_around -= 1.0;
				float cb_next = (A+B+fixed[curt*3+c])/pixel_around;//(pixel_around*ct - (nt+wt+st+et) + (nb+wb+sb+eb))/pixel_around;
				
				buf2[3*curt+c] = min(255.0, max(0.0, cb_next));
			}
		}
		else
		  {
			  for(int c = 0; c < 3; c++) {
				  buf2[3*curt+c] = bg[3*curt+c];
			  }
		  }



	}
}

__global__ void CalculateFixed(
		const float *background,
		const float *target,
		const float *mask,
		float *fixed,
		float *strictInteriorPixels,
		float *bg,
		const int wb, const int hb, const int wt, const int ht,
		const int oy, const int ox
		)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	//if (yt < ht and xt < wt and mask[curt] < 127.0f) {
	if (yt < ht && xt < wt) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if(0 <= yb && yb < hb && 0 <= xb && xb < wb) {
				bg[curt*3+0] = background[curb*3+0];
				bg[curt*3+1] = background[curb*3+1];
				bg[curt*3+2] = background[curb*3+2];
		}
		if(mask[curt] > 127.0f) {
			int curtN = wt*(yt-1)+xt;
			int curtW = wt*yt+(xt-1);
			int curtS = wt*(yt+1)+xt;
			int curtE = wt*yt+(xt+1);
			int flag = 0; //0 = interior pixel, 1 = border pixel
			if(yt == 0 || mask[curtN] < 127.0f) flag = 1;
			if(xt == 0 || mask[curtW] < 127.0f) flag = 1;
			if(yt == ht-1 || mask[curtS] < 127.0f) flag = 1;
			if(xt == wt-1 || mask[curtE] < 127.0f) flag = 1;
			strictInteriorPixels[curt] = (flag == 1) ? 0 : 1;
			
			for(int c = 0; c < 3; c++) {
				float C = 0.0f;
				if(yt > 0) C += target[3*curt+c] - target[3*curtN+c];
				if(xt > 0) C += target[3*curt+c] - target[3*curtW+c];
				if(yt < ht-1) C += target[3*curt+c] - target[3*curtS+c];
				if(xt < wt-1) C += target[3*curt+c] - target[3*curtE+c];
				fixed[3*curt+c] = C;
			}
		}
		/*else {
		//for(int c = 0; c < 3; c++) {
		//if(target[curt*3+c] > 0.0f) 
		fixed[curt*3+0] = target[curt*3+0];
		fixed[curt*3+1] = target[curt*3+1];
		fixed[curt*3+2] = target[curt*3+2];
		//else
		//	fixed[curt*3+c] = 0.0f;

		//	}
		}*/
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
	float *fixed, *buf1, *buf2, *strictInteriorPixels, *bg;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));
	cudaMalloc(&bg, 3*wt*ht*sizeof(float));
	cudaMalloc(&strictInteriorPixels, wt*ht*sizeof(float));


	
	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);

	CalculateFixed<<<gdim, bdim>>>(
			background, target, mask, fixed, strictInteriorPixels, bg,
			wb, hb, wt, ht, oy, ox
			);
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

	//cudaMemset(buf1, 0, sizeof(float)*3*wt*ht);
	cudaMemset(buf2, 0, sizeof(float)*3*wt*ht);

	for(int i = 0; i < 100000; i++) {
		PoissonImageCloningIteration<<<gdim, bdim>>>(
				fixed, mask, target, strictInteriorPixels, bg, buf1, buf2, wt, ht
				);
		PoissonImageCloningIteration<<<gdim, bdim>>>(
				fixed, mask, target, strictInteriorPixels, bg, buf2, buf1, wt, ht
				);
	}

	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);

	//cudaMemset(buf1, 0, sizeof(float)*3*wt*ht);
	//cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
	SimpleClone<<<gdim, bdim>>>(
			background, buf1, mask, output,
			wb, hb, wt, ht, oy, ox
			);

	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);

}

