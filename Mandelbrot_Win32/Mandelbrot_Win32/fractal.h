#include <GL/glut.h>
#include <cuda_runtime.h>
#include <stdio.h>

typedef void (*func) (double a, double b, double c, double d, double *x, double *y);

//enum FractalRenderType { FRACTAL_RENDERTYPE_HISTOGRAM, FRACTAL_RENDERTYPE_EXPONENTIAL };

__global__ void mandelNum ( int *counts, double *data, long *histogram, const int imgWidth, const int imgHeight, const int depth, const double midX, const double midY,
							const double scale, const int iterations, const bool julia, const double juliaX, const double juliaY );

__global__ void coloring( int *counts, double *data, long *histogram, GLubyte *array, const int imgWidth, const int imgHeight, const int depth, const int iterations, const int samplingFactor );

__global__ void mandelNumExp( GLubyte *array, const int imgWidth, const int imgHeight, const int depth, const double midX, const double midY,
							  const double scale, const int iterations, const bool julia, const double juliaX, const double juliaY );

__global__ void partialSum( long *input, const int length );


__device__ void mul( double a, double b, double c, double d, double *x, double *y );
__device__ void add( double a, double b, double c, double d, double *x, double *y );
__device__ void pow( double a, double b, double n, double *x, double *y );
__device__ void exp( double a, double b, double *x, double *y );


void CalcFractal( GLubyte *devArray, int *counts, double *data, long *histogram, double dPosX, double dPosY, double dScale, int iWidth, int iHeight, 
				  int iDepth, int iIterations, bool bIsJulia, double dJuliaX, double dJuliaY);