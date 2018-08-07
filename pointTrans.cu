// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
// Thread block size
#define BLOCK_SIZE 16
// Matrix dimensions
#define WA 4 // Matrix A width
#define HA 4 // Matrix A height
#define SIZE_A WA*HA
#define HB WA  // Matrix B height
//#define WC WB  // Matrix C width 
#define HC HA // Matrix C height
#define MAX_VALUE 10
#define TRANSLATION 1
#define SCALING 2
// includes, project
#include <cutil.h>

// includes, kernels
#include "matrixMul_kernel.cu"
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
float *create_matB( int, int, float* ); 
void printDiff(float*, float*, int, int);
float *tran_mat(float, float, float, float*);
float *sca_mat(float, float, float, float *);
void intiate_matA(float *);
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    runTest(argc, argv);

    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv){
    CUT_DEVICE_INIT();

    // set seed for rand()
    srand(2018);
	int WB=0;
	printf("Informe o numero de pontos para serem gerados randomicamente: ");
	scanf("%d", &WB);
	int WC = WB;											
    // allocate host memory for matrices A and B
    unsigned int mem_size_A = sizeof(float) * SIZE_A;
    float* h_A = (float*) calloc(mem_size_A);				
    unsigned int size_B = WB * HB;								
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);

    // initialize host memory
    intiate_matA(h_A);		
    h_B = create_matB(WB,HB,h_B);
	// allocate device memory
    float* d_A;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_A, mem_size_A));
    float* d_B;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_B, mem_size_B));

    // copy host memory to device
    CUDA_SAFE_CALL(cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL(cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice) );

    // allocate device memory for result
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* d_C;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_C, mem_size_C));

    // allocate host memory for the result
    float* h_C = (float*) malloc(mem_size_C);
    
    // create and start timer
    unsigned int timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    CUT_SAFE_CALL(cutStartTimer(timer));

    // setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(WC / threads.x);						// HC / threads  = 0 talvez HC % threads ? 	grid unidimencional

    // execute the kernel
    matrixMul<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);			// só coluna é passada

    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

    // copy result from device to host
    CUDA_SAFE_CALL(cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost) );

    // stop and destroy timer
    CUT_SAFE_CALL(cutStopTimer(timer));
    printf("Processing time: %f (ms) \n", cutGetTimerValue(timer));
    CUT_SAFE_CALL(cutDeleteTimer(timer));

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    CUDA_SAFE_CALL(cudaFree(d_A));
    CUDA_SAFE_CALL(cudaFree(d_B));
    CUDA_SAFE_CALL(cudaFree(d_C));
}

// Allocates a matrix with random float entries.
float *create_matB( int width, int height, float *pMat) {

	//float *pMat = (float *)malloc( width * height * sizeof( float ) );
	srand(time(NULL));

	int i;
	for (i = 0 ; i < width*height; i++ ){
		if (i < width*(height-1))
			v[i] = (float)(rand() % MAX_VALUE);
		else
			v[i] = 1.0;
	}	
	return pMat;
}

float *tran_mat(float tx, float ty, float tz, float *v){
	int i;
	
	for (i = 0 ; i < SIZE_A; i++ ){
		if(!i%5)
			v[i] = 1;			    
	}
	v[WA-1] = tx;     v[2*WA-1] = ty;     v[3*WA-1] = tz;

	return(v); 
}

float *sca_mat(float sx, float sy, float sz, float *v){
	v[0] = sx;
	v[2*WA+1] = sy;
	v[3*WA+2] = sz;
	v[SIZE_A-1] = 1;

	return(v); 
}

void intiate_matA(float *pMat){
	int op;
	float x, y,z;
	
	printf("\nInforme o Tipo de Operação: \n1- Translation\n2- Scaling\n\nNumero da operação: ");
    scanf("%d", &op);
    
    switch(op){
        case TRANSLATION:
						printf("\nInforme o valor de Tx : ");
						scanf("%f", &x);
						printf("Informe o valor de Ty : ");
						scanf("%f", &y);
						printf("Informe o valor de Tz : ");
						scanf("%f", &z);
                        tran_mat(x,y,z,pMat);
                        break;
        case SCALING:
						printf("\nInforme o valor de Sx : ");
						scanf("%f", &x);
						printf("Informe o valor de Sy : ");
						scanf("%f", &y);
						printf("Informe o valor de Sz : ");
						scanf("%f", &z);
                        sca_mat(x,y,z,pMat);
                        break;
    }
}

















