/*
 * nn.c
 *
 *  Created on: 5 jul. 2016
 *      Author: ecesar
 *
 *      Descripció:
 *      Xarxa neuronal simple de tres capes. La d'entrada que són els pixels d'una
 *      imatge ppm (mirar descripció del format al comentari de readImg) de 50x50 (un total de 7500
 *      entrades). La capa oculta amb un nombre variable de neurones (amb l'exemple proporcionat 60
 *      funciona relativament bé, però si incrementem el nombre de patrons d'entrada caldrà variar-lo).
 *      Finalment, la capa de sortida (que ara té 6 neurones ja que només l'entrenem per reconèixer 6
 *      patrons, si es vol extendre caldrà augmentar aquest nombre).
 *      El programa passa per una fase d'entrenament en la qual processa un conjunt de patrons (en
 *      l'exemple proporcionat són 30 amb les lletres A, a, B, b, C, c, D, d, E, e, F i f, fent servir
 *      diferents fonts i mides). Un cop ha calculat els pesos entre la capa d'entrada i l'oculta i entre
 *      aquesta i la de sortida, passa a la fase de reconéixament, on llegeix 17 patrons d'entrada
 *      (es proporcionen exemples per aquests patrons), i intenta reconèixer de quina lletre es tracte.
 */

/*******************************************************************************
*    Aquest programa és una adaptació del fet per (el link a la pàgina i codi original el teniu a l'enunciat):
*
*    nn.c   1.0                                       � JOHN BULLINARIA  2004  *
*******************************************************************************/

/*      To compile use "cc nn.c -O -lm -o nn" and then run using "./nn"       */
/*      For explanations see:  http://www.cs.bham.ac.uk/~jxb/NN/nn.html       */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <string.h>
#include "nn-vo.h"
#include "common.h"
#include "cuda.h"

#define FULL_MASK 0xffffffff


			/*for( int j = 0 ; j < NUMHID ; j++ ) {    // compute hidden unit activations
				float SumH = 0.0;
				for( int i = 0 ; i < NUMIN ; i++ ) SumH += tSet[p][i] * WeightIH[j][i];
				Hidden[p][j] = 1.0/(1.0 + exp(-SumH)) ;
			}*/
__global__
void ComputeHidden(int p, float* WeightIH_d, char* tSet_d, float* Hidden_d){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ float H_dd[1024];
		if(i<NUMHID*NUMIN){
			H_dd[threadIdx.x] = tSet_d[p*NUMIN+i%NUMIN] * WeightIH_d[i];

			for (unsigned int stride = blockDim.x/2; stride > 0;  stride /= 2){ //First col will have the valid Hidden_d's values
				__syncthreads();
				if (threadIdx.x < stride)
					H_dd[threadIdx.x] += H_dd[threadIdx.x+stride];
			}

			__syncthreads();
			if(threadIdx.x == 0){
				atomicAdd(&Hidden_d[p*NUMHID+i/NUMIN], H_dd[threadIdx.x]);
			}
		}
}


__global__
void ComputeHiddenSigma(int p, float* Hidden_d){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i<NUMHID)
		Hidden_d[p*NUMHID+i] = 1.0/(1.0 + exp(-Hidden_d[p*NUMHID+i]));
}


__global__
void ComputeOutput(int p, float* O_d, float* Hidden_d, float* WeightHO_d, float* Output_d){
	int i = threadIdx.x + blockDim.x*blockIdx.x; //Block Dimx = 128
	int j = threadIdx.y + blockDim.y*blockIdx.y; //Block Dimy = 6;

	if(i<NUMHID && j<NUMOUT){
		O_d[j*128+i] = Hidden_d[p*NUMHID+i%NUMHID] * WeightHO_d[j*NUMHID+i]; //O_d is 128 values long to not lose values when doing the reduction

		for (unsigned int stride = blockDim.x/2; stride > 0;  stride /= 2){//First col will have the valid Hidden_d's values
			__syncthreads();
			if (threadIdx.x < stride)
				O_d[j*128+i] += O_d[j*128+i+stride];
		}
		__syncthreads();
		if(threadIdx.x == 0){
			atomicAdd(&Output_d[p*NUMOUT+(j*NUMHID+i)/NUMHID], O_d[j*128+i]);
		}
	}
}

__global__
void ComputeOutputSigma(int p, float* Output_d){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i<NUMOUT)
		Output_d[p*NUMOUT+i] = 1.0/(1.0 + exp(-Output_d[p*NUMOUT+i]));
}

__global__
void ComputeError(int p, float* E_d, float* Error_d, float* Target_d, float* Output_d){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i<NUMOUT){
		E_d[i] = 0.5 * (Target_d[p*NUMOUT+i] - Output_d[p*NUMOUT+i]) * (Target_d[p*NUMOUT+i] - Output_d[p*NUMOUT+i]);
		atomicAdd(Error_d, E_d[i]);
	}
}
__global__
void ComputeDeltaO(int p, float* DeltaO_d, float* Target_d, float* Output_d){
	int i = threadIdx.x + blockDim.x*blockIdx.x;

	if(i<NUMOUT)
		DeltaO_d[i] = (Target_d[p*NUMOUT+i] - Output_d[p*NUMOUT+i]) * Output_d[p*NUMOUT+i] * (1.0 - Output_d[p*NUMOUT+i]);

}

__global__
void ComputeDeltaH(int p, float* WeightHO_d, float* DeltaO_d, float* DeltaH_d, float* Hidden_d){
	int i = threadIdx.x + blockDim.x*blockIdx.x;

	if(i<NUMOUT*NUMHID){
		DeltaH_d[i%NUMHID] = 0;
		__syncthreads();
		float a = WeightHO_d[i] * DeltaO_d[i/NUMHID];
		atomicAdd(&DeltaH_d[i%NUMHID], a);
	}
	__syncthreads();
	if(i<NUMHID)
			DeltaH_d[i] = DeltaH_d[i] * Hidden_d[p*NUMHID+i] * (1.0 - Hidden_d[p*NUMHID+i]);
}

__global__
void UpdateWeightIH(int p, float* DeltaWeightIH_d, float* WeightIH_d, char* tSet_d, float* DeltaH_d, float eta, float alpha){
	int i = threadIdx.x + blockDim.x*blockIdx.x;

	if(i<NUMHID*NUMIN){
		DeltaWeightIH_d[i] = eta * tSet_d[p*NUMIN+i%NUMIN] * DeltaH_d[i/NUMIN] + alpha * DeltaWeightIH_d[i];
		WeightIH_d[i] += DeltaWeightIH_d[i] ;
	}

}

__global__
void UpdateWeightHO(int p, float* DeltaWeightHO_d, float* WeightHO_d, float* Hidden_d, float* DeltaO_d, float eta, float alpha){
	int i = threadIdx.x + blockDim.x*blockIdx.x;

	if(i<NUMOUT*NUMHID){
		DeltaWeightHO_d[i] = eta * Hidden_d[p*NUMHID+i%NUMHID] * DeltaO_d[i/NUMHID] + alpha * DeltaWeightHO_d[i];
		WeightHO_d[i] += DeltaWeightHO_d[i] ;
	}

}


void printFromGPU(float* address, float* address_d, int elements, int size){
	cudaMemcpy(address, address_d, elements*size, cudaMemcpyDeviceToHost);
	for(int i = 0; i< elements; i++)
		printf("%f ", address[i]);
	printf("\n");
}


void freeDeltaWeights(float *DeltaWeightIH[], float *DeltaWeightHO[]){
	for( int i = 0; i < NUMHID; i++)
		free(DeltaWeightIH[i]);
	for( int i = 0; i < NUMOUT; i++)
		free(DeltaWeightHO[i]);
}

void freeWeights(float *WeightIH[],  float *WeightHO[]){
	for( int i = 0; i < NUMHID; i++)
		free(WeightIH[i]);
	for( int i = 0; i < NUMOUT; i++)
		free(WeightHO[i]);
}

void freeFList( int nf, char *flist[] ){
	for (int i = 0; i < nf; i++) free(flist[i]);
}

void freeTSet( int np, char **tset ){
	for( int i = 0; i < np; i++ ) free( tset[i] );
	free(tset);
}

/*
 * En primer lloc  el número màgic P6 (char *), després un comentari (al haver estat generades amb gimp),
 * després l'amplada de la imatge (char *), l'alçada (char *) i el valor màxim de color (char *),
 * després  un salt de línia i comença la llista de pixels.
 * Cada pixel está compost per tres bytes (valors RGB) codificats en binari.
 */
char *readImg( char *fname ){

	FILE *fd;
	char saux[100];
	int width, heigth, waste;
	char *img;


	if( (fd = fopen( fname, "rb" )) == NULL) return NULL;
	fscanf(fd,"%s\n", saux);
	if( strcmp("P6", saux) ) return NULL;
	fgets(saux,100,fd);
	fscanf(fd,"%d %d\n %d\n", &width, &heigth, &waste);

	img = (char *)malloc(width*heigth*3);
	printf("Reading Img %s (%d,%d) %d\n", fname, width, heigth, fread(img, 1, width*heigth*3, fd));
	fclose(fd);
	return img;
}

char **loadTrainingSet(int nf, char *ifileL[]){

	char **tset;

	tset = (char **)malloc(nf*sizeof(char *));
	int error = 0;

	for (int i = 0; i < nf; i++){
		if ((tset[i] = readImg( ifileL[i] )) == NULL) error = 1;
	}

	if (error) return NULL;
	return tset;
}

void trainN(){

	char **tSet;
	char *fname[NUMPAT];

	initFileList( fname );

	if( (tSet = loadTrainingSet(NUMPAT,fname)) == NULL){
		printf("Error!!\n");
		exit(-1);
	}

	float smallwt = 0.1;

	for( int i = 0; i < NUMHID; i++){
		if ((WeightIH[i] = (float *)malloc((NUMIN)*sizeof(float))) == NULL){
			printf("Out of Mem\n");
			exit(-1);
		}
	}
	for(int j = 0; j < NUMIN; j++)
		for( int i = 0; i < NUMHID; i++){
			WeightIH[i][j] = 2.0 * ( rando() + 0.5 ) * smallwt;
		}


	for( int i = 0; i < NUMOUT; i++){
		if ((WeightHO[i] = (float *)malloc((NUMHID)*sizeof(float))) == NULL){
			printf("Out of Mem\n");
			exit(-1);
		}

	}
	for(int j = 0; j < NUMHID; j++)
		for( int i = 0; i < NUMOUT; i++){
			WeightHO[i][j] = 2.0 * ( rando() + 0.5 ) * smallwt;
		}

    float Error, eta = 0.2, alpha = 0.5;
	int ranpat[NUMPAT];
	float Output[NUMPAT][NUMOUT];

	int cuda_err;

    /*********************************************************************************
    *MALLOCS IN DEVICE
    *********************************************************************************/
	/*And some memcpy of constant values*/

    float* Hidden_d;
    cuda_err = cudaMalloc((void**)&Hidden_d, NUMPAT*NUMHID*sizeof(float));
    if(cuda_err) printf("Hidden_d Malloc err\n");

    //float SumH;
    float* SumH_d;
    cuda_err = cudaMalloc((void**)&SumH_d, sizeof(float));
    if(cuda_err) printf("SumH_d Malloc err\n");

    /*WeightIH*/
    float* WeightIH_d;
    cuda_err = cudaMalloc((void**)&WeightIH_d, NUMHID*NUMIN*sizeof(float));
    if(cuda_err) printf("WeightIH_d Malloc err\n");

    /*tSet_d*/
    char* tSet_d;
    cuda_err = cudaMalloc((void**)&tSet_d, NUMPAT*NUMIN*sizeof(char));
    if(cuda_err) printf("tSet_d Malloc err\n");
    cuda_err = 0;
    for(int i = 0; i<NUMPAT; i++)
			cuda_err+=cudaMemcpy(&tSet_d[i*NUMIN], tSet[i], NUMIN*sizeof(char), cudaMemcpyHostToDevice);
    if(cuda_err) printf("tSet_d Memcpy err\n");

    float* Output_d;
    cuda_err = cudaMalloc((void**)&Output_d, NUMPAT*NUMOUT*sizeof(float));
    if(cuda_err) printf("Output_d Malloc err\n");

    float* WeightHO_d;
    cuda_err = cudaMalloc((void**)&WeightHO_d, NUMOUT*NUMHID*sizeof(float));
    if(cuda_err) printf("WeightHO_d Malloc err\n");

    float* E_d; //Error aux array
    cuda_err = cudaMalloc((void**)&E_d, NUMOUT*sizeof(float));
    if(cuda_err) printf("E_d Malloc err\n");

    float* Error_d;
    cuda_err = cudaMalloc((void**)&Error_d, sizeof(float));
    if(cuda_err) printf("Error_d Malloc err\n");

    float* Target_d;
    cuda_err = cudaMalloc((void**)&Target_d, NUMPAT*NUMOUT*sizeof(float));
    if(cuda_err) printf("Target_d Malloc err\n");
    cuda_err = 0;
    for(int i = 0; i<NUMPAT; i++)
    	cuda_err += cudaMemcpy(&Target_d[i*NUMOUT], Target[i], NUMOUT*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_err) printf("Target_d Memcpy err\n");

    float* O_d;
    cuda_err = cudaMalloc((void**)&O_d, NUMOUT*128*sizeof(float));
    if(cuda_err) printf("O_d Memcpy err\n");

    float* DeltaO_d;
    cuda_err = cudaMalloc((void**)&DeltaO_d, NUMOUT*sizeof(float));
    if(cuda_err) printf("DeltaO_d Memcpy err\n");

    float* DeltaH_d;
    cuda_err = cudaMalloc((void**)&DeltaH_d, NUMHID*sizeof(float));
    if(cuda_err) printf("DeltaH_d Memcpy err\n");

    float* DeltaWeightIH_d;
    cuda_err = cudaMalloc((void**)&DeltaWeightIH_d, NUMHID*NUMIN*sizeof(float));
    if(cuda_err) printf("DeltaWeightIH_d Memcpy err\n");

    float* DeltaWeightHO_d;
    cuda_err = cudaMalloc((void**)&DeltaWeightHO_d, NUMOUT*NUMHID*sizeof(float));
    if(cuda_err) printf("DeltaWeightHO_d Memcpy err\n");

    /*********************************************************************************
    *END MALLOCS
    *********************************************************************************/

    /* INIT WEIGHTS ON DEVICE */
    for(int j = 0; j< NUMHID; j++)
    	cuda_err += cudaMemcpy(&WeightIH_d[j*NUMIN], WeightIH[j], NUMIN*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_err) printf("WeightIH_d Memcpy err\n");

    for(int j = 0; j< NUMOUT; j++)
    	cuda_err += cudaMemcpy(&WeightHO_d[j*NUMHID], WeightHO[j], NUMHID*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_err) printf("WeightHO_d Memcpy err\n");

    /*********************************************************************************
    *COMPUTE
    *********************************************************************************/
    for( int epoch = 0 ; epoch < 1000000 ; epoch++) {    // iterate weight updates
        for( int p = 0 ; p < NUMPAT ; p++ )   // randomize order of individuals
            ranpat[p] = p ;
        for( int p = 0 ; p < NUMPAT ; p++) {
            int np = rand()%NUMPAT;
            int op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
        }
        Error = 0.0 ;
        cudaMemcpy(Error_d, &Error, sizeof(float), cudaMemcpyHostToDevice);

        printf("."); fflush(stdout);

        for( int np = 0 ; np < NUMPAT ; np++ ) {    // repeat for all the training patterns
        	//float DeltaO[NUMOUT], DeltaH[NUMHID];
        	int p = ranpat[np];


			/*___________HIDDEN____________*/
			ComputeHidden<<<ceil(NUMHID*NUMIN/1024.0), 1024>>>(p, WeightIH_d, tSet_d, Hidden_d);
			ComputeHiddenSigma<<<ceil(NUMHID/1024.0), NUMHID>>>(p, Hidden_d);

			/*___________OUTPUT____________*/
			/*Compute Output*/
			dim3 nThreads(128,NUMOUT,1);
			ComputeOutput<<<ceil(1), nThreads>>>(p, O_d, Hidden_d, WeightHO_d, Output_d);
			ComputeOutputSigma<<<ceil(1), NUMOUT>>>(p, Output_d);

			/*Computer Error*/
			ComputeError<<<ceil(NUMOUT/1024.0), NUMOUT>>>(p, E_d, Error_d, Target_d, Output_d);
			cudaMemcpy(&Error, Error_d, sizeof(float), cudaMemcpyDeviceToHost);
			if(cuda_err) printf("Error_d Memcpy err\n");

			/*Compute Delta Output*/
			ComputeDeltaO<<<ceil(NUMOUT/1024.0), NUMOUT>>>(p, DeltaO_d, Target_d, Output_d);

			/* ______________WeightIH____________*/
			/*Compute DeltaH*/

			ComputeDeltaH<<<ceil(1), 512>>>(p, WeightHO_d, DeltaO_d, DeltaH_d, Hidden_d);

			UpdateWeightIH<<<ceil(NUMHID*NUMIN/1024.0), 1024>>>(p, DeltaWeightIH_d, WeightIH_d, tSet_d, DeltaH_d, eta, alpha);

			/* ______________WeightHO____________*/

			UpdateWeightHO<<<ceil(1), 512>>>(p, DeltaWeightHO_d, WeightHO_d, Hidden_d, DeltaO_d, eta, alpha);


        }
        if( !(epoch%100) ) printf("\nEpoch %-5d :   Error = %f \n", epoch, Error) ;
        if( Error < 0.0004 ) {
        	printf("\nEpoch %-5d :   Error = %f \n", epoch, Error) ; break ;  // stop learning when 'near enough'
        }
    }
    /*********************************************************************************
    *END COMPUTE
    *********************************************************************************/

    /*********************************************************************************
    *COPY FINAL VALUES TO HOST
    *********************************************************************************/
    /*WeightIH*/
    cuda_err = 0;
    for(int j = 0; j< NUMHID; j++)
    	cuda_err += cudaMemcpy(WeightIH[j], &WeightIH_d[j*NUMIN], NUMIN*sizeof(float), cudaMemcpyDeviceToHost);
    if(cuda_err) printf("WeightIH_d Memcpy to H err\n");

    /*WeightHO*/
    cuda_err = 0;
    for(int j = 0; j< NUMOUT; j++)
    	cuda_err += cudaMemcpy(WeightHO[j], &WeightHO_d[j*NUMHID], NUMHID*sizeof(float), cudaMemcpyDeviceToHost);
    if(cuda_err) printf("WeightHO_d Memcpy to H err\n");

	/*Output*/
	cuda_err = 0;
	for(int p = 0; p<NUMOUT; p++)
		cuda_err += cudaMemcpy(Output[p], &Output_d[p*NUMOUT], NUMOUT*sizeof(float), cudaMemcpyDeviceToHost);
	if(cuda_err) printf("Output_d Memcpy to H err\n");

    //freeDeltaWeights(DeltaWeightIH, DeltaWeightHO);
	freeFList( NUMPAT, fname );
	freeTSet( NUMPAT, tSet );

    for( int p = 0 ; p < NUMPAT ; p++ ) {
    	printf("\n%d\t", p);
        for( int k = 0 ; k < NUMOUT ; k++ ) {
            printf("%f\t%f\t", Target[p][k], Output[p][k]);
        }
    }
    printf("%c", tSet[2][2]); //avoid optimize out
    printf("\n");
}

void printRecognized(int p, float Output[]){
	int imax = 0;

	for( int i = 1; i < NUMOUT; i++)
		if ( Output[i] > Output[imax] )
			imax = i;
	printf("El patró %d és una %c\t", p, 'A' + imax);
    for( int k = 0 ; k < NUMOUT ; k++ )
        printf("\t%f\t", Output[k]) ;
    printf("\n");

}

void runN(){
	char **rSet;
	char *fname[NUMRPAT];

	initRunFileList( fname );

	if( (rSet = loadTrainingSet(NUMRPAT,fname)) == NULL){
		printf("Error!!\n");
		exit(-1);
	}

	float Hidden[NUMHID], Output[NUMOUT];

    for( int p = 0 ; p < NUMRPAT ; p++ ) {    // repeat for all the recognition patterns
        for( int j = 0 ; j < NUMHID ; j++ ) {    // compute hidden unit activations
            float SumH = 0.0;
            for( int i = 0 ; i < NUMIN ; i++ )
		SumH += rSet[p][i] * WeightIH[j][i];
            Hidden[j] = 1.0/(1.0 + exp(-SumH)) ;
        }

        for( int k = 0 ; k < NUMOUT ; k++ ) {    // compute output unit activations
            float SumO = 0.0;
            for( int j = 0 ; j < NUMHID ; j++ )
		SumO += Hidden[j] * WeightHO[k][j] ;
            Output[k] = 1.0/(1.0 + exp(-SumO)) ;   // Sigmoidal Outputs
        }
        printRecognized(p, Output);
    }

	freeFList(NUMRPAT, fname);
	freeTSet( NUMRPAT, rSet );
}

int main() {
	clock_t start = clock();
	// srand(start); 		//Comentat porta a resultats deterministes (però en el cas real ha d'aparéixer)
	trainN();
	runN();

	freeWeights(WeightIH, WeightHO);
	clock_t end = clock();
	printf("\n\nGoodbye! (%f sec)\n\n", (end-start)/(1.0*CLOCKS_PER_SEC)) ;

    return 0;
}

/*******************************************************************************/
