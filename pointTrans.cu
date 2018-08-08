#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
//#include <iostream>
//#include <iomanip>
//#include <cutil.h>
//#include <matrixMul_kernel.cu>	// includes, kernels		

#define BLOCK_SIZE 16

#define WA 4
#define HA 4
#define SIZE_A WA*HA
#define HB WA 					// Matrix B height
#define HC HA 					// Matrix C height
#define MAX_VALUE 10
#define TRANSLATION 1
#define SCALING 2

//kernel defs
#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) CUT_BANK_CHECKER(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) CUT_BANK_CHECKER(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif

// declaration, forward
void runTest(int argc, char** argv);
float *create_matB( int, int, float* ); 
void printDiff(float*, float*, int, int);
float *tran_mat(float, float, float, float*);
float *sca_mat(float, float, float, float *);
void intiate_matA(float *);


/*-------------------------------------------------------------------/
			Matrix multiplication on the device: C = A * B
				wA is A's width and wB is B's width
-------------------------------------------------------------------*/

__global__ void
matrixMul( float* C, float* A, float* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + wA * ty + tx];
        BS(ty, tx) = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}



/*-------------------------------------------------------------------/
							Program main
-------------------------------------------------------------------*/
int main(int argc, char** argv)
{
	runTest(argc, argv);

}


void runTest(int argc, char** argv){

	srand(2018);
	int WB=0;
	printf("Informe o numero de pontos para serem gerados randomicamente: ");
	scanf("%d", &WB);
	int WC = WB;		
									
	// allocate host memory for matrices A and B
	unsigned int mem_size_A = sizeof(float) * SIZE_A;
	float* h_A = (float*) calloc(SIZE_A, mem_size_A);				
	unsigned int size_B = WB * HB;								
	unsigned int mem_size_B = sizeof(float) * size_B;
	float* h_B = (float*) malloc(mem_size_B);

	// initialize host memory
	intiate_matA(h_A);		
	h_B = create_matB(WB,HB,h_B);

	// allocate device memory
	float* d_A;
	cudaMalloc((void**) &d_A, mem_size_A);
	float* d_B;
	cudaMalloc((void**) &d_B, mem_size_B);

	// copy host memory to device
	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

	// allocate device memory for result
	unsigned int size_C = WC * HC;
	unsigned int mem_size_C = sizeof(float) * size_C;
	float* d_C;
	cudaMalloc((void**) &d_C, mem_size_C);

	// allocate host memory for the result
	float* h_C = (float*) malloc(mem_size_C);
	
	// create and start timer
	//alguma coisa aqui??

	// setup execution parameters
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(WC / threads.x);

	// execute the kernel
	matrixMul<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);

	// copy result from device to host
	cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	// stop and destroy timer
	// outra coisa aqui??

	// clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}


// Allocates a matrix with random float entries.
float *create_matB( int width, int height, float *pMat) {

	//float *pMat = (float *)malloc( width * height * sizeof( float ) );
	srand(time(NULL));

	int i;
	for (i = 0 ; i < width*height; i++ ){
		if (i < width*(height-1))
			pMat[i] = (float)(rand() % MAX_VALUE);
		else
			pMat[i] = 1.0;
	}	
	return pMat;
}

float *tran_mat(float tx, float ty, float tz, float *v){
	int i;
	
	for (i = 0 ; i < SIZE_A; i++ ){
		if(!i%5)
			v[i] = 1;				
	}
	v[WA-1] = tx;	 v[2*WA-1] = ty;	 v[3*WA-1] = tz;

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

