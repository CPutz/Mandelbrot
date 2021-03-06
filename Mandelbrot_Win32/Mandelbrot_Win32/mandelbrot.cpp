#define GLEW_STATIC

#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <Windows.h>

#include "mandelbrot.h"
#include "cuda_helper.h"
#include "fractal.h"


Mandelbrot::Mandelbrot( int iWidth, int iHeight) {
	this->dPosX = 0;
	this->dPosY = 0;
	this->iWidth = iWidth;
	this->iHeight = iHeight;
	this->iDepth = 4;
	this->iSize = this->iWidth * this->iHeight * this->iDepth * sizeof(GLubyte);
	this->dScale = 1;
	this->bIsJulia = false;
	this->bIsFinished = true;
	this->bIsFlushed = false;
}


void Mandelbrot::Initialize() {
	glewInit();

	createBuffer( &this->buffer, this->iSize);
	createTexture( &this->tex, this->iWidth, this->iHeight );
	createHistogram( 128 );

	cudaMalloc( ( void** ) &this->devCounts, this->iWidth * this->iHeight * sizeof( int ) );
	cudaMalloc( ( void** ) &this->devData,   this->iWidth * this->iHeight * sizeof( double ) * 2 );
	cudaMalloc( ( void** ) &this->devArray,  this->iWidth * this->iHeight * this->iDepth * sizeof( GLubyte ) );
	cudaMalloc( ( void** ) &this->devCalcArray, this->iWidth * this->iHeight * this->iDepth * sizeof( GLubyte ) );
}


void Mandelbrot::Resize(int width, int height) {

	while( !this->bIsFinished );

	this->iWidth = width;
	this->iHeight = height;
	this->iSize = this->iWidth * this->iHeight * this->iDepth * sizeof(GLubyte);

	cudaFree( ( void** ) &this->devArray );
	cudaFree( ( void** ) &this->devCalcArray );
	cudaFree( ( void** ) &this->devCounts );
	cudaFree( ( void** ) &this->devData );
	cudaFree( ( void** ) &this->devHistogram );

	checkCudaErrors( cudaGLUnregisterBufferObject( this->buffer ), __LINE__, true );
	glDeleteBuffers( 1, &this->buffer );
    glDeleteTextures( 1, &this->tex );

	
	createBuffer( &this->buffer, this->iSize);
	createTexture( &this->tex, this->iWidth, this->iHeight );

	cudaMalloc( ( void** ) &this->devCounts, this->iWidth * this->iHeight * sizeof( int ) );
	cudaMalloc( ( void** ) &this->devData,   this->iWidth * this->iHeight * sizeof( double ) * 2 );
	cudaMalloc( ( void** ) &this->devArray,  this->iWidth * this->iHeight * this->iDepth * sizeof( GLubyte ) );
	cudaMalloc( ( void** ) &this->devCalcArray, this->iWidth * this->iHeight * this->iDepth * sizeof( GLubyte ) );
}



void Mandelbrot::Update(int iterations) {
	
	if (this->bIsFinished) {

		if ( this->iOldIterations != iterations ) {
			createHistogram(iterations);
			this->iOldIterations = iterations;
		}
	
		this->bIsFinished = false;
		this->bIsFlushed = false;

		boost::thread(&Mandelbrot::calculate, this, iterations);

		//calculate(iterations);
	}
}


void Mandelbrot::calculate(int iterations) {

	//checkCudaErrors( cudaGLMapBufferObject( ( void** ) &this->devArray, this->buffer ), __LINE__, false );

	CalcFractal( this->devCalcArray, this->devCounts, this->devData, this->devHistogram, this->dPosX, this->dPosY, this->dScale, this->iWidth, this->iHeight, this->iDepth, iterations, this->bIsJulia, this->dJuliaX, this->dJuliaY );
	
    //checkCudaErrors( cudaGLUnmapBufferObject( this->buffer ), __LINE__, false );

	this->bIsFinished = true;
}


void Mandelbrot::WriteBuffer() {

	checkCudaErrors( cudaGLMapBufferObject( ( void** ) &this->devArray, this->buffer ), __LINE__, false );

	cudaMemcpy( this->devArray, this->devCalcArray, this->iSize, cudaMemcpyDeviceToDevice );

	checkCudaErrors( cudaGLUnmapBufferObject( this->buffer ), __LINE__, false );

	this->bIsFlushed = true;
}


void Mandelbrot::createHistogram(int iterations)
{
	if ( this->devHistogram )
		cudaFree( ( void** ) &this->devHistogram );
	
	cudaMalloc( ( void** ) &this->devHistogram, (iterations + 1) * sizeof( long ) );
}


void Mandelbrot::createBuffer( GLuint* b, int size ) {
    glGenBuffers( 1, b );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, *b );
	glBufferData( GL_PIXEL_UNPACK_BUFFER, size,  0, GL_DYNAMIC_COPY );
	
    checkCudaErrors( cudaGLRegisterBufferObject( *b ), __LINE__, true );
}


void Mandelbrot::createTexture( GLuint* texture, int iWidth, int iHeight ) {
	glEnable( GL_TEXTURE_2D );
    glGenTextures( 1, texture );
    glBindTexture( GL_TEXTURE_2D, *texture );
    {
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_BORDER );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_BORDER );

        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, iWidth, iHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
    }
    glBindTexture( GL_TEXTURE_2D, 0 );
    glDisable( GL_TEXTURE_2D );
}


Mandelbrot::~Mandelbrot() {
	cudaFree( ( void** ) &this->devArray );
	cudaFree( ( void** ) &this->devCalcArray );
	cudaFree( ( void** ) &this->devCounts );
	cudaFree( ( void** ) &this->devData );
	cudaFree( ( void** ) &this->devHistogram );
	
	checkCudaErrors( cudaGLUnregisterBufferObject( this->buffer ), __LINE__, true );
	glDeleteBuffers( 1, &this->buffer );
    glDeleteTextures( 1, &this->tex );
}


void Mandelbrot::setPosition( double x, double y ) {
	this->dPosX = x;
	this->dPosY = y;
}

void Mandelbrot::setScale( double dScale ) {
	this->dScale = dScale;
}

bool Mandelbrot::getIsJulia() {
	return this->bIsJulia;
}

void Mandelbrot::setIsJulia( bool val ) {
	this->bIsJulia = val;
}

void Mandelbrot::setJulia( double x, double y ) {
	this->dJuliaX = x;
	this->dJuliaY = y;
}

GLuint Mandelbrot::getTexture() {
	return this->tex;
}

GLuint Mandelbrot::getBuffer() {
	return this->buffer;
}

bool Mandelbrot::isFinished() {
	return this->bIsFinished;
}

bool Mandelbrot::isFlushed() {
	return this->bIsFlushed;
}


