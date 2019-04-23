
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h> 

#pragma warning(disable:4996)

#define Imax 1024
#define Jmax 1024
#define Nmax Imax*Jmax

const int iter_num = 40000;
double  rho_node[Imax][Jmax];

__constant__ double  tau = 1.0;
__constant__ double  T = 0.560, kappa = 0.010, a = 9.00 / 49.00, b = 2.00 / 21.00;

cudaError_t device_main();
__global__ void initialize_func(double *rho_node, double *u_node, double *v_node, double *random);
__global__ void getf_eq(double *rho_node, double *u_node, double *v_node, double *f_eq, double *f_temp);
__global__ void getf(double *f, double *f_eq, double *f_temp);
__global__ void stream_func(double *f, double *f_temp);
__global__ void collision_func(double *rho_node, double *f_temp);
void output(int counter);

int main()
{
	int numDevices;
	cuInit(0);
	cuDeviceGetCount(&numDevices);
	printf("%d devices detected:\n", numDevices);
	for (int i = 0; i< numDevices; i++){
		char szName[256];
		CUdevice device;
		cuDeviceGet(&device, i);
		cuDeviceGetName(szName, 255, device);
		printf("\t%s\n", szName);
	}
	for (int i = 0; i<numDevices; i++){
		struct cudaDeviceProp device_prop;
		if (cudaGetDeviceProperties(&device_prop, i) == cudaSuccess){
			printf("device properties is :\n"
			"\t device name is %s\n"
			"\t totalGlobalMem is %d\n"
			"\t sharedMemPerBlock is %d\n"
			"\t regsPerBlock is %d\n"
			"\t warpSize is %d\n"
			"\t memPitch is %d\n"
			"\t maxThreadsPerBlock is %d\n"
			"\t maxThreadsDim [3] is %d X %d X %d\n"
			"\t maxGridSize [3] is %d X %d X %d\n"
			"\t totalConstMem is %d\n"
			"\t device version is major %d ,minor %d\n"
			"\t clockRate is %d\n"
			"\t textureAlignment is %d\n"
			"\t deviceOverlap is %d\n"
			"\t multiProcessorCount is %d\n",
			device_prop.name,
			device_prop.totalGlobalMem,
			device_prop.sharedMemPerBlock,
			device_prop.regsPerBlock,
			device_prop.warpSize,
			device_prop.memPitch,
			device_prop.maxThreadsPerBlock,
			device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2],
			device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2],
			device_prop.totalConstMem,
			device_prop.major, device_prop.minor,
			device_prop.clockRate,
			device_prop.textureAlignment,
			device_prop.deviceOverlap,
			device_prop.multiProcessorCount);
		}
	}
	
	cudaError_t cudaStatus = device_main();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Device main Failed!");
		system("pause");
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		system("pause");
		return 1;
	}

	system("pause");
	return 0;
}

//##############################  kernel begin ######################################//
cudaError_t device_main()
{
	cudaError_t cudaStatus;
	clock_t start, finish;
	double random[Imax][Jmax];

	srand(1234);

	for (int j = 0; j < Jmax; ++j)
		for (int i = 0; i < Imax; ++i)
			random[j][i] = (double)rand() / (double)(RAND_MAX);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) printf("\nCudaSetDevice Wrong!");

	//############### (1)申请显存 ###############
	printf("\nApply for memory in GPU.\n");
	double *dev_f, *dev_f_temp, *dev_f_eq;
	cudaMalloc((void **)&dev_f, Imax*Jmax * 9 * sizeof(double));
	cudaMalloc((void **)&dev_f_temp, Imax*Jmax * 9 * sizeof(double));
	cudaMalloc((void **)&dev_f_eq, Imax*Jmax * 9 * sizeof(double));

	double *dev_rho_node, *dev_u_node, *dev_v_node;
	cudaMalloc((void **)&dev_rho_node, Imax*Jmax*sizeof(double));
	cudaMalloc((void **)&dev_u_node, Imax*Jmax*sizeof(double));
	cudaMalloc((void **)&dev_v_node, Imax*Jmax*sizeof(double));

	double *dev_random;
	cudaMalloc((void **)&dev_random, Imax*Jmax*sizeof(double));
	cudaStatus = cudaMemcpy(dev_random, random, Imax*Jmax*sizeof(double), cudaMemcpyHostToDevice);

	int thread_dim = 16;
	dim3 dim_block = dim3(thread_dim, thread_dim, 1);
	dim3 dim_grid = dim3(Imax / thread_dim, Jmax / thread_dim, 1);
	printf("\nGrid Dimension:grid.x=%d, grid.y=%d", dim_grid.x, dim_grid.y);
	printf("\nBlock Dimension:block.x=%d, block.y=%d\n", dim_block.x, dim_block.y);

	//############### (2) 初始化 ############### 
	start = clock();
	//printf("\n(1)Initial");
	initialize_func << < dim_grid, dim_block >> > (dev_rho_node, dev_u_node, dev_v_node, dev_random);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Initial kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaDeviceSynchronize();
	cudaMemcpy(rho_node, dev_rho_node, Imax*Jmax*sizeof(double), cudaMemcpyDeviceToHost);
	output(0);

	//############### (4) 迭代求解 ############### 
	for (int counter = 0; counter <iter_num; counter++) {
		if ( (int)fmod((double)counter, 1000) == 0) printf("\nStep:%6d", counter);
		//printf("\n%d-(2)Getf_eq", counter);
		getf_eq << < dim_grid, dim_block >> > (dev_rho_node, dev_u_node, dev_v_node, dev_f_eq, dev_f_temp);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Getf_eq kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaDeviceSynchronize();

		//printf("\n%d-(3)Getf", counter);
		getf << < dim_grid, dim_block >> > (dev_f, dev_f_eq, dev_f_temp);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Stream kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaDeviceSynchronize();

		//printf("\n%d-(4)Stream", counter);
		stream_func << < dim_grid, dim_block >> > (dev_f, dev_f_temp);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Stream kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaDeviceSynchronize();

		//printf("\n%d-(5)Collision", counter);
		collision_func << < dim_grid, dim_block >> > (dev_rho_node, dev_f_temp);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Collision kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaDeviceSynchronize();

		if ((int)fmod((double)counter, 1000) == 0){
			cudaMemcpy(rho_node, dev_rho_node, Imax*Jmax*sizeof(double), cudaMemcpyDeviceToHost);
			output(counter);
		}
	}
	printf("\nIteration over!\n");

	cudaMemcpy(rho_node, dev_rho_node, Imax*Jmax*sizeof(double), cudaMemcpyDeviceToHost);
	finish = clock();
	printf("\nSpent %.3f seconds in calculating.\n", (double)(finish - start) / CLOCKS_PER_SEC);
	output(iter_num);

	//############### (5) 释放显存 ############### 
Error:
	cudaFree(dev_f);
	cudaFree(dev_f_temp);
	cudaFree(dev_f_eq);
	cudaFree(dev_rho_node);
	cudaFree(dev_u_node);
	cudaFree(dev_v_node);
	cudaFree(dev_random);

	return cudaStatus;
}
//##############################  kernel end ######################################//

__global__ void initialize_func(double *rho_node, double *u_node, double *v_node, double *random)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ind = i + j * Imax;

	rho_node[ind] = 3.0 + random[ind];
	u_node[ind] = 0.0;
	v_node[ind] = 0.0;

	if (j >= Jmax || i >= Imax) {
		printf("index cross the border!");
		return;
	}
}

__global__ void getf_eq(double *rho_node, double *u_node, double *v_node, double *f_eq, double *f_temp)
{
	int  i_w, i_e, j_n, j_s;
	double  rho_xgrad, rho_ygrad, rho_laplace;
	double  A0, A1, A2, B1, B2, C0, C1, C2, D1, D2, \
		Gxx1, Gxx2, Gyy1, Gyy2, Gxy1, Gxy2, P, vel_node;
	double  vel_dire[8];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ind = i + j * Imax;


	P = rho_node[ind] * T / (1.0 - rho_node[ind] * b) - a * pow(rho_node[ind], 2);

	if (i != 0)
		i_w = i - 1;
	else
		i_w = Imax - 1;

	if (i != Imax - 1)
		i_e = i + 1;
	else
		i_e = 0;

	if (j != 0)
		j_n = j - 1;
	else
		j_n = Jmax - 1;

	if (j != Jmax - 1)
		j_s = j + 1;
	else
		j_s = 0;

	int ind_e = i_e + j * Imax;
	int ind_w = i_w + j * Imax;
	int ind_s = i + j_s * Imax;
	int ind_n = i + j_n * Imax;
	int ind_en = i_e + j_n * Imax;
	int ind_wn = i_w + j_n * Imax;
	int ind_ws = i_w + j_s * Imax;
	int ind_es = i_e + j_s * Imax;

	rho_xgrad = (rho_node[ind_e] - rho_node[ind_w]) / 2.00;
	rho_ygrad = (rho_node[ind_n] - rho_node[ind_s]) / 2.00;
	rho_laplace = (rho_node[ind_e] + rho_node[ind_n] + rho_node[ind_w] + rho_node[ind_s])*4.0 / 6.0 + \
		(rho_node[ind_en] + rho_node[ind_wn] + rho_node[ind_ws] + rho_node[ind_es]) / 6.00 - \
		20.0 / 6.0*rho_node[ind];

	A1 = 1.0 / 3.0*(P - kappa*rho_node[ind] * rho_laplace);
	A2 = A1 / 4.0;
	A0 = rho_node[ind] - 5.0 / 3.0*(P - kappa*rho_node[ind] * rho_laplace);
	B2 = rho_node[ind] / 12.0;
	B1 = 4.0*B2;
	C2 = -rho_node[ind] / 24.0;
	C1 = 4.0*C2;
	C0 = -2.0*rho_node[ind] / 3.0;
	D1 = rho_node[ind] / 2.0;
	D2 = rho_node[ind] / 8.0;

	Gxx1 = kappa / 4.0*(pow(rho_xgrad, 2) - pow(rho_ygrad, 2));
	Gxx2 = Gxx1 / 4.0;
	Gyy1 = -Gxx1;
	Gyy2 = -Gxx2;
	Gxy1 = kappa / 2.0*rho_xgrad*rho_ygrad;
	Gxy2 = Gxy1 / 4.0;

	int index_1 = ind + 1 * Nmax;
	int index_2 = ind + 2 * Nmax;
	int index_3 = ind + 3 * Nmax;
	int index_4 = ind + 4 * Nmax;
	int index_5 = ind + 5 * Nmax;
	int index_6 = ind + 6 * Nmax;
	int index_7 = ind + 7 * Nmax;
	int index_8 = ind + 8 * Nmax;

	u_node[ind] = ((f_temp[index_1] - f_temp[index_3]) + (f_temp[index_5] - f_temp[index_7]) + (f_temp[index_8] - f_temp[index_6])) / rho_node[ind];
	v_node[ind] = ((f_temp[index_2] - f_temp[index_4]) + (f_temp[index_5] - f_temp[index_7]) + (f_temp[index_6] - f_temp[index_8])) / rho_node[ind];
	vel_node = u_node[ind] * u_node[ind] + v_node[ind] * v_node[ind];

	vel_dire[0] =  u_node[ind];
	vel_dire[1] =  v_node[ind];
	vel_dire[2] = -u_node[ind];
	vel_dire[3] = -v_node[ind];
	vel_dire[4] =  u_node[ind] + v_node[ind];
	vel_dire[5] = -u_node[ind] + v_node[ind];
	vel_dire[6] = -u_node[ind] - v_node[ind];
	vel_dire[7] =  u_node[ind] - v_node[ind];
	
	f_eq[ind] = A0 + C0*vel_node;
	f_eq[index_1] = A1 + B1*vel_dire[0] + C1*vel_node + \
		D1*vel_dire[0] * vel_dire[0] + Gxx1;
	f_eq[index_2] = A1 + B1*vel_dire[1] + C1*vel_node + \
		D1*vel_dire[1] * vel_dire[1] + Gyy1;
	f_eq[index_3] = A1 + B1*vel_dire[2] + C1*vel_node + \
		D1*vel_dire[2] * vel_dire[2] + Gxx1;
	f_eq[index_4] = A1 + B1*vel_dire[3] + C1*vel_node + \
		D1*vel_dire[3] * vel_dire[3] + Gyy1;
	f_eq[index_5] = A2 + B2*vel_dire[4] + C2*vel_node + \
		D2*vel_dire[4] * vel_dire[4] + Gxx2 + Gyy2 + 2.0*Gxy2;
	f_eq[index_6] = A2 + B2*vel_dire[5] + C2*vel_node + \
		D2*vel_dire[5] * vel_dire[5] + Gxx2 + Gyy2 - 2.0*Gxy2;
	f_eq[index_7] = A2 + B2*vel_dire[6] + C2*vel_node + \
		D2*vel_dire[6] * vel_dire[6] + Gxx2 + Gyy2 + 2.0*Gxy2;
	f_eq[index_8] = A2 + B2*vel_dire[7] + C2*vel_node + \
		D2*vel_dire[7] * vel_dire[7] + Gxx2 + Gyy2 - 2.0*Gxy2;

	/*if (i == 64 & j == 64) {
		printf("\nGetf_eq: dir=%d, f_eq[%d]=%f, f_temp[%d]=%f", 0, ind, f_eq[ind], ind, f_temp[ind]);
		printf("\nGetf_eq: dir=%d, f_eq[%d]=%f, f_temp[%d]=%f", 1, index_1, f_eq[index_1], index_1, f_temp[index_1]);
		printf("\nGetf_eq: dir=%d, f_eq[%d]=%f, f_temp[%d]=%f", 2, index_2, f_eq[index_2], index_2, f_temp[index_2]);
		printf("\nGetf_eq: dir=%d, f_eq[%d]=%f, f_temp[%d]=%f", 3, index_3, f_eq[index_3], index_3, f_temp[index_3]);
		printf("\nGetf_eq: dir=%d, f_eq[%d]=%f, f_temp[%d]=%f", 4, index_4, f_eq[index_4], index_4, f_temp[index_4]);
		printf("\nGetf_eq: dir=%d, f_eq[%d]=%f, f_temp[%d]=%f", 5, index_5, f_eq[index_5], index_5, f_temp[index_5]);
		printf("\nGetf_eq: dir=%d, f_eq[%d]=%f, f_temp[%d]=%f", 6, index_6, f_eq[index_6], index_6, f_temp[index_6]);
		printf("\nGetf_eq: dir=%d, f_eq[%d]=%f, f_temp[%d]=%f", 7, index_7, f_eq[index_7], index_7, f_temp[index_7]);
		printf("\nGetf_eq: dir=%d, f_eq[%d]=%f, f_temp[%d]=%f", 8, index_8, f_eq[index_8], index_8, f_temp[index_8]);
	}*/

}

__global__ void getf(double *f, double *f_eq, double *f_temp)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ind = i + j * Imax;

	for (int dir = 0; dir < 9; dir++){
		int index = ind + dir*Nmax;
		f[index] = f_temp[index] - 1.0 / tau * (f_temp[index] - f_eq[index]);
	}
}

__global__ void stream_func(double *f, double *f_temp)
{
	int  i_w, i_e, j_n, j_s;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ind = i + j * Imax;

	if (i != 0)
		i_w = i - 1;
	else
		i_w = Imax - 1;

	if (i != Imax - 1)
		i_e = i + 1;
	else
		i_e = 0;

	if (j != 0)
		j_n = j - 1;
	else
		j_n = Jmax - 1;

	if (j != Jmax - 1)
		j_s = j + 1;
	else
		j_s = 0;

	int ind_e = i_e + j * Imax;
	int ind_w = i_w + j * Imax;
	int ind_s = i + j_s * Imax;
	int ind_n = i + j_n * Imax;
	int ind_en = i_e + j_n * Imax;
	int ind_wn = i_w + j_n * Imax;
	int ind_ws = i_w + j_s * Imax;
	int ind_es = i_e + j_s * Imax;
	int ind_new;

	for (int dir = 0; dir < 9; dir++) {
		if (dir == 0) ind_new = ind;
		else if (dir == 1)  ind_new = ind_e;
		else if (dir == 2) ind_new = ind_n;
		else if (dir == 3) ind_new = ind_w;
		else if (dir == 4) ind_new = ind_s;
		else if (dir == 5) ind_new = ind_en;
		else if (dir == 6) ind_new = ind_wn;
		else if (dir == 7) ind_new = ind_ws;
		else if (dir == 8) ind_new = ind_es;
		
		int index = ind + dir*Nmax;
		int index_new = ind_new + dir*Nmax;
		f_temp[index_new] = f[index];
	}
}

__global__ void collision_func(double *rho_node, double *f_temp)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ind = i + j * Imax;

	rho_node[ind] = 0.0;
	for (int dir = 0; dir < 9; dir++) {
		int index = ind + dir*Nmax;
		rho_node[ind] = rho_node[ind] + f_temp[index];
	}
}

void output(int counter)
{
	char name1[50];
	char tec_title[50], field[100];

	int  i, j;

	FILE *fp1;

	sprintf_s(name1, 50, "result\\%-6dtwophase_swift.plt", counter);

	//strcpy_s(name1, 20, "twophase_swift.plt");
	fopen_s(&fp1, name1, "w+");
	fputs("VARIABLES=\"X\",\"Y\",\"density\"\n", fp1);
	sprintf_s(tec_title, 50, "ZONE I=%d,J=%d,F=POINT\n", Imax, Jmax);
	fputs(tec_title, fp1);
	//write(10, "(A7,I3,A3,I3,A8)") "ZONE I=", Imax, ",J=", Jmax, ",F=POINT";

	for (i = 0; i < Imax; ++i)
	{
		for (j = 0; j < Jmax; ++j)
		{
			sprintf_s(field, 100, "%4d %4d %17.9f\n", i, Jmax - 1 - j, rho_node[j][i]);
			fputs(field, fp1);
		}
	}
	fclose(fp1);

}
