//-----------------------------------------
// Autor: Farias
// Data : Nov 2010
// Goal : Multiply two matrices in GPU
//-----------------------------------------

/***************************************************************************************************
	Includes
***************************************************************************************************/

#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>

#include "rf-time.h"


/***************************************************************************************************
	Defines
***************************************************************************************************/

#define ERROR_ALOCATING_MATRIX "Erro: na alocacao de memoria para matriz."
#define ERROR_INVALID_PARMS    "Erro: invalid parameters."
#define ERROR_NULL_VECTOR      "Erro: tentativa de imprimir vetor nulo."

#define CUDA_SAFE_CALL 
#define ELEM(l,c,WID) ((c)+(l)*(WID))


/***************************************************************************************************
	Functions
***************************************************************************************************/

using namespace std;


/**************************************************************************************************/

__host__ void erro( const char tipoDeErro[] ) {

	fprintf( stderr, "%s\n", tipoDeErro );
	exit(0);

}


/**************************************************************************************************/
int *createMatrix( int width, int height ) {

	int *pMat = (int *)malloc( width * height * sizeof( int ) );
	if( !pMat ) {
		erro( ERROR_ALOCATING_MATRIX );
	}

	//srand( get_clock_sec() );

	for( int i = 0 ; i < width*height ; i++ ) {

		//pMat[ i ] = my_rand( numeroDeElementos * 1000 );
		pMat[ i ] = 1;

	}
	
	return pMat;

}


/**************************************************************************************************/
void printMatrix( int *mat, int width, int height ) {

	int l, c;
	
	if( !mat ) {
		erro( ERROR_NULL_VECTOR );
	}

	cout << "      ";
	for( c = 0 ; c < width ; c++ ) 
		cout << setw(7) << c;
	cout << endl;
	for( l = 0 ; l < height ; l++ ) {
	
		cout << setw(6) << l;
		for( c = 0 ; c < width ; c++ ) 
			cout << setw(7) << mat[ ELEM( l, c, width ) ];
		cout << endl;

	}
	
}


/**************************************************************************************************/
bool compareMatrices( int *matA, int * matB, int width, int height ) {

	int l, c;

	for( l = 0 ; l < height ; l++ ) 
	
		for( c = 0 ; c < width ; c++ ) 

			if( matA[ ELEM( l, c, width ) ] != matB[ ELEM( l, c, width ) ] )
				return false;

	return true;

}


/**************************************************************************************************/
__host__ void multMatricesCPU( int *matA, int * matB, int *matC, int width, int height ) {

	int l, c, k, sum;

	for( l = 0 ; l < height ; l++ ) {
	
		for( c = 0 ; c < width ; c++ ) {

			sum = 0;

			for( k = 0 ; k < width ; k++ )

				sum += matA[ ELEM( l, k, width ) ] * matB[ ELEM( k, c, width ) ];
				
			matC[ ELEM( l, c, width ) ] = sum;

		}

	}

}


/**************************************************************************************************/
__global__ void multMatricesGPU( int n, int *matA, int *matB, int *matC ) {

	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int l = blockIdx.y*blockDim.y + threadIdx.y;

	if( l < n && c < n ) {
		
		int sum = 0;
		
		for( int k = 0 ; k < n ; k++ ) {
			
			sum += matA[ ELEM( l, k, n ) ]*matB[ ELEM( k, c, n ) ];
			
		}
		
		matC[ ELEM( l, c, n ) ] = sum;

	}
	
}



void deviceQuery( void ) {

	int deviceCount = 0;
	if( cudaGetDeviceCount( &deviceCount ) != cudaSuccess ) {
		cout << "cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n";
		cout << "Aborting...\n";
		exit( 0 );
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if( deviceCount == 0 ) {
		cout << "There is no device supporting CUDA\n";
		exit( 0 );
	}

	for( int dev = 0; dev < deviceCount ; ++dev ) {
		
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties( &deviceProp, dev );

		if( dev == 0 ) {
			
			// This function call returns 9999 for both major & minor 
			// fields, if no CUDA capable devices are present
			if( deviceProp.major == 9999 && deviceProp.minor == 9999 )
				cout << "There is no device supporting CUDA.\n";
			else if( deviceCount == 1 )
				cout << "There is 1 device supporting CUDA\n";
			else
				cout << "There are "<<  deviceCount << " devices supporting CUDA\n";

		}
		printf( "Device %d: \"%s\"\n", dev, deviceProp.name );

		printf( "  CUDA Capability Major revision number:         %d\n", deviceProp.major);
		printf( "  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);
		printf( "  Total amount of global memory:                 %u bytes\n", \
			(unsigned int)deviceProp.totalGlobalMem );
#if CUDART_VERSION >= 2000
		printf( "  Number of multiprototalGlobalMemcessors:                     %d\n", deviceProp.multiProcessorCount);
		//printf("  Number of cores:                               %d\n", nGpuArchCoresPerSM[deviceProp.major] * deviceProp.multiProcessorCount);
#endif

	}

} 



/**************************************************************************************************/
__host__ int main( int argc, char *argv[] ) {

	int blSizeX = 32, blSizeY = 32;
	double start_time, cpu_mult_time, gpu_mult_time, copy_to_time, copy_from_time;
	
	deviceQuery();

	// Neste exemplo, consideramos apenas matrizes quadradas
	// Ou seja, o parametro N de entrada indicara uma matrix NxN

	// Trata parâmetros de entrada
	int h_numElem = 3;

	if( argc > 1 ) {
		h_numElem = atoi( argv[ 1 ] );
		if( h_numElem < 1 ) {
			erro( ERROR_INVALID_PARMS );
		}
	}

	switch( argc ) {

	case 3:
		blSizeX = blSizeY = atoi( argv[ 2 ] );
		break;
	case 4:
		blSizeX = atoi( argv[ 2 ] );
		blSizeY = atoi( argv[ 3 ] );
	}
	cout << "Multiplicar duas matrizes " << h_numElem << "x" << h_numElem << endl;

	// Gera vetorA e vetorB
	cout << "Cria matrix A\n";
	int *h_matA = createMatrix( h_numElem, h_numElem );
	cout << "Cria Matrix B\n";
	int *h_matB = createMatrix( h_numElem, h_numElem );
	// Alocar matriz resultado
	int *h_matC = (int*)malloc( h_numElem*h_numElem*sizeof( int ) );
	if( !h_matC ) {
		erro( ERROR_ALOCATING_MATRIX );
	}
	int *h_gpuResp = (int*)malloc( h_numElem*h_numElem*sizeof( int ) );
	if( !h_gpuResp ) {
		erro( ERROR_ALOCATING_MATRIX );
	}


	/*
	// Imprime vetorA
	cout << "Matrix A -----------------------------------------------------" << endl;
	printMatrix( h_matA, h_numElem, h_numElem );

	// Imprime vetorB
	cout << "Matrix B -----------------------------------------------------" << endl;
	printMatrix( h_matB, h_numElem, h_numElem );
	


	// Imprime resultado CPU
	cout << "Matrix Resultado ---------------------------------------------" << endl;
	printMatrix( h_matC, h_numElem, h_numElem );
	*/

	cout << "Alocando matrizes A, B, Resp\n";

	// Aloca memória no device e copia vetorA e vetorB para lá
	int* d_matA;
	int* d_matB;
	int* d_matC;
	//CUDA_SAFE_CALL( cudaMalloc( (void**)&d_matA, h_numElem*h_numElem*sizeof( int ) ) );
	//CUDA_SAFE_CALL( cudaMalloc( (void**)&d_matB, h_numElem*h_numElem*sizeof( int ) ) );
	//CUDA_SAFE_CALL( cudaMalloc( (void**)&d_matC, h_numElem*h_numElem*sizeof( int ) ) );

	cudaMalloc( (void**)&d_matA, h_numElem*h_numElem*sizeof( int ) );
	cudaMalloc( (void**)&d_matB, h_numElem*h_numElem*sizeof( int ) );
	cudaMalloc( (void**)&d_matC, h_numElem*h_numElem*sizeof( int ) );

	cout << "Copiando matrizes A, B\n";

	start_time = get_clock_msec();
	CUDA_SAFE_CALL( cudaMemcpy( d_matA, h_matA, h_numElem*h_numElem*sizeof( int ), cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy( d_matB, h_matB, h_numElem*h_numElem*sizeof( int ), cudaMemcpyHostToDevice ) );
	copy_to_time = get_clock_msec() - start_time;

	// Calcula dimensoes da grid e dos blocos
	dim3 blockSize( blSizeX, blSizeY, 1 );

	int numBlocosX = h_numElem / blockSize.x + ( h_numElem % blockSize.x == 0 ? 0 : 1 );
	int numBlocosY = h_numElem / blockSize.y + ( h_numElem % blockSize.y == 0 ? 0 : 1 );
	dim3 gridSize( numBlocosX, numBlocosY, 1 );

	cout << "Multiplicando em GPU..... \n";
	
	// Chama SomarVetoresGPU
	start_time = get_clock_msec();
	multMatricesGPU<<< gridSize, blockSize >>>( h_numElem, d_matA, d_matB, d_matC );
	cudaThreadSynchronize();
	gpu_mult_time = get_clock_msec() - start_time;

	cout << " Done" << endl;

	// Copia o resultado de volta para o host
	start_time = get_clock_msec();
	CUDA_SAFE_CALL( cudaMemcpy( h_gpuResp, d_matC, h_numElem*h_numElem*sizeof( int ), cudaMemcpyDeviceToHost ) );
	copy_from_time = get_clock_msec() - start_time;


	// Imprime tempos
	cout << "--------------------------------------------------------------" << endl;
	cout << "Informacoes da execucao..." << endl;
	cout << "--------------------------------------------------------------" << endl;
	cout << "Dimensoes das Matrizes: " << h_numElem << "x" << h_numElem << endl;
	cout << "\tBlock: (" << blSizeX << "," << blSizeY << ")" << endl;
	cout << "\tGrid : (" << numBlocosX << "," << numBlocosY << ")" << endl;
	cout << "Tempos de execucao: " << endl;
	cout << "\tCopia CPU->GPU das matrizes A e B: " << copy_to_time << endl;
	cout << "\tCopia GPU->CPU da matriz resultado: " << copy_from_time << endl;
	cout << "\tGPU multiplicacao: " << gpu_mult_time << endl;

	/*
	// Imprime resultado GPU
	cout << "Matrix Resultado ---------------------------------------------" << endl;
	printMatrix( h_gpuResp, h_numElem, h_numElem );
	*/

	if( h_numElem <= 512 ) {
		// Calcula multiplicacao na cpu
		start_time = get_clock_msec();
		multMatricesCPU( h_matA, h_matB, h_matC, h_numElem, h_numElem );
		cpu_mult_time = get_clock_msec() - start_time;
		
		cout << "\tCPU multiplicacao: " << cpu_mult_time << endl;
		
		cout << "--------------------------------------------------------------" << endl;
		if( compareMatrices( h_matC, h_gpuResp, h_numElem, h_numElem ) )
			cout << "Resultado CORRETO  :-)" << endl;
		else
			cout << "Resultado INCORRETO!!!!!!!!!!!!!!!!!!!  :-(" << endl;
	}
	cout << "--------------------------------------------------------------" << endl;


	// Libera memória do device
	CUDA_SAFE_CALL( cudaFree( d_matA ) );
	CUDA_SAFE_CALL( cudaFree( d_matB ) );
	CUDA_SAFE_CALL( cudaFree( d_matC ) );
	
	// Libera memória do host
	free( h_matA );
	free( h_matB );
	free( h_matC );
	free( h_gpuResp );

	return 0;

}
