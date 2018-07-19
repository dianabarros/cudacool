#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX_VALUE 10
#define TRANSLATION 1
#define ROTATION 2
#define SCALING 3
#define DIM 4


clock_t t;

void start(){
	t = clock();
}

void stop(){
	t = clock() -t;
	double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
	printf("function took %f miliseconds to execute \n", time_taken*1000);
}

/* aloca memoria para as matrizes  */
float **Alocar_matriz_real(int m, int n)
{
	float **v;  /* ponteiro para a matriz */
	int   i;    /* variavel auxiliar      */

	if (m < 1 || n < 1) { /* verifica parametros recebidos */
		printf ("** Erro: Parametro invalido **\n");
		return (NULL);
	}

	/* aloca as linhas da matriz */
	v = calloc (m, sizeof(float *));	/* Um vetor de m ponteiros para float */
	if (v == NULL) {
		printf ("** Erro: Memoria Insuficiente **");
		return (NULL);
	}

	/* aloca as colunas da matriz */
	for ( i = 0; i < m; i++ ) {
		v[i] =  calloc (n, sizeof(float));	/* m vetores de n floats */
		if (v[i] == NULL) {
			printf ("** Erro: Memoria Insuficiente **");
			return (NULL);
		}
	}
	return (v); /* retorna o ponteiro para a matriz */
}

/* Libera memoria ocupada pelas matrizes */
float **Liberar_matriz_real (int m, int n, float **v)
{
	int  i;  /* variavel auxiliar */

	if (v == NULL)
	   	return (NULL);

	if (m < 1 || n < 1) {  /* verifica parametros recebidos */
		printf ("** Erro: Parametro invalido **\n");
		return (v);
	}

	for (i = 0; i < m; i++)
	   	free (v[i]); /* libera as linhas da matriz */

	free (v);      /* libera a matriz (vetor de ponteiros) */

	return (NULL); /* retorna um ponteiro nulo */
}

float **tran_mat(int tx, int ty, int tz, float **v)
{
	int i, j;
	//~ tx,ty,tz;

	//~ printf("\nInforme o valor de Tx : ");
	//~ scanf("%d", &tx);
	//~ printf("Informe o valor de Ty : ");
	//~ scanf("%d", &ty);
	//~ printf("Informe o valor de Tz : ");
	//~ scanf("%d", &tz);
	for (i = 0 ; i < DIM; i++ )
		for (j = 0; j < DIM-1; j++) {
		    if( i==j)
			    v[i][j] = 1;
			//~ else
			    //~ v[i][j] = 0;
			    
		}
	v[0][DIM-1] = tx;     v[1][DIM-1] = ty;     v[2][DIM-1] = tz; v[3][DIM-1] = 1;
	printf("\n");

	return(v); 
}

float **sca_mat(int sx, int sy, int sz, float **v)
{
	int i, j;
	//~ ,sx,sy,sz;

	//~ printf("\nInforme o valor de Sx : ");
	//~ scanf("%d", &sx);
	//~ printf("Informe o valor de Sy : ");
	//~ scanf("%d", &sy);
	//~ printf("Informe o valor de Sz : ");
	//~ scanf("%d", &sz);
	for (i = 0 ; i < DIM; i++ )
		for (j = 0; j < DIM; j++) {
		    if( i==j)
			    v[i][j] = 1;
			//~ else
			    //~ v[i][j] = 0;
			    
		}
	v[0][0] = sx;     v[1][1] = sy;     v[2][2] = sz;
	printf("\n");

	return(v); 
}

/* Essa funcao faz a multiplicacao entre as matrize.
 * Retorna um matriz com resultado da multplicao
 * */
float **mult(int ma, int mb , int na, int nb, float **a, float **b, float **mr)
{
	int i, j , v;
	start();
	for (i = 0 ; i < ma; i++ )
		for (j = 0; j < nb; j++){
		    mr[i][j] = 0;
			for (v = 0; v < ma; v++){
				mr[i][j] = mr[i][j] + a[i][v] * b[v][j];
			}
		}
	stop();
	return(mr);
}

/* funcao pra imprimir na tela as matrizes  */
void imprime(int ma, int mb , int na, int nb , float **a, float **b, float **mr)
{
	int i, j;

	/* Impressao das Matrizes */
	printf("\nMATRIX A:\n");
	for (i = 0; i < ma; i++) {
		for ( j = 0; j < na; j++)
			printf("%f ", a[i][j]);
		printf("\n");
	}
	printf("MATRIX B:\n");
	for (i = 0; i < mb; i++) {
		for ( j = 0; j < nb; j++)
			printf("%f ", b[i][j]);
	printf("\n");
	}

	printf("MATRIX C:\n");
	for (i = 0; i < ma ; i++) {
		for ( j = 0; j < nb ; j++)
			printf("%f", mr[i][j]);
		printf("\n");
	}
}

float **rand_mat(int m, int n, float **v)
{
	int i, j;
	srand(time(NULL));
	
	for (i = 0 ; i < m; i++ )
		for (j = 0; j < n; j++) {
			//~ printf("Posicao a%d%d \n", i+1, j+1);
			//~ scanf("%f", &v[i][j]);
			if (i < m-1)
				v[i][j] = (float)(rand() % MAX_VALUE);
			else
				v[i][j] = 1.0;
		}
	return(v); 
}

float **rotx_mat(double angle, float **v){
	int i, j;
	for (i = 0 ; i < DIM; i++ )
		for (j = 0; j < DIM; j++) {
			if (i == j){
				if ( i == 0 || i == DIM-1)
					v[i][j] = 1.0;
				else
					v[i][j] = cos(angle);
			}
			if ( i + j == DIM -1)
				if (i == 1 || j == 1) //só funciona pra DIM 4
					v[i][j] = (i-j)*sin(angle);
		}
	
		return(v);
}

float **roty_mat(double angle, float **v){
	int i, j;
	//só funciona pra DIM 4
	for (i = 0 ; i < DIM; i++ )
		for (j = 0; j < DIM; j++) {
			if (i == j){
				if ( i % 2 == 0)
					v[i][j] = cos(angle);
				else
					v[i][j] = 1.0;
			}
			else
				if ( i + j == 2)
					v[i][j] = ((j-i)/2)*sin(angle);
		}
	
		return(v);
}

float **rotz_mat(double angle, float **v){
		int i, j;
	//só funciona pra DIM 4
	for (i = 0 ; i < DIM; i++ )
		for (j = 0; j < DIM; j++) {
			if (i == j){
				if ( i < 2)
					v[i][j] = cos(angle);
				else
					v[i][j] = 1.0;
			}
			else
				if ( i + j == 1)
					v[i][j] = (i-j)*sin(angle);
		}
	
		return(v);
}

/*
 * Essa é a funcao principal
 */
int main(int argc, char **argv)
{
	float **A;  /* matriz a ser alocada */
	float **B;  /* matriz a ser alocada */
	float **MR;  /* matriz a ser alocada */
	int la =DIM, lb=DIM, ca=DIM, cb=0;   /* numero de linhas e colunas da matriz */
    int op, x, y,z;
	
	if (argc>1){
		if((cb=atoi(*(++argv))))
			printf("Numero de pontos para serem gerados randomicamente: %d\n",cb);
		else{
			printf("Informação Invalida!\n");
			argc = 1;		
		}
	}
	if (argc ==1){
		printf("Informe o numero de pontos para serem gerados randomicamente: ");
		scanf("%d", &cb);
	}

	/* Chama a funcao para alocar a matriz */
	A = Alocar_matriz_real (la, ca);
	B = Alocar_matriz_real (lb, cb);
	MR = Alocar_matriz_real (la, cb);
	
	printf("\nInforme o Tipo de Operação: \n1- Translation\n2- Rotation\n3- Scaling\n\nNumero da operação: ");
    scanf("%d", &op);
    
    switch(op){
        case TRANSLATION:
						printf("\nInforme o valor de Tx : ");
						scanf("%d", &x);
						printf("Informe o valor de Ty : ");
						scanf("%d", &y);
						printf("Informe o valor de Tz : ");
						scanf("%d", &z);
                        A = tran_mat(x,y,z,A);
                        break;
        case ROTATION:
                        A = rotz_mat(1.5707,A); //teste com 1.5707 rad
                        break;
        case SCALING:
						printf("\nInforme o valor de Sx : ");
						scanf("%d", &x);
						printf("Informe o valor de Sy : ");
						scanf("%d", &y);
						printf("Informe o valor de Sz : ");
						scanf("%d", &z);
                        A = sca_mat(x,y,z,A);
                        break;
    }

	B = rand_mat(lb, cb, B);

	/* chama a funcao pra fazer muultiplicacao das matrizes */
	MR = mult(la, lb, ca, cb, A, B, MR);

	/* chama a funcao pra mostrar na tela o resultado da multiplicacao */
	imprime(la, lb , ca, cb, A, B, MR);
	
	/* desaloca a memoria, nao mais nescessaria */
	A = Liberar_matriz_real (la, ca, A);
	B = Liberar_matriz_real (lb, cb, B);
	MR = Liberar_matriz_real (la, cb, MR);

	return 0;
} /* fim do programa */
