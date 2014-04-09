#include <cuda_runtime.h>

#include <iostream>
#include <cstdlib>


void checkCudaErrors( cudaError_t code, int line, bool abort ) {
	if( code != cudaSuccess ) {
        std::cout << "CUDA error: " << cudaGetErrorString( code ) << ". line: " << line << std::endl;
        if( abort )
            exit( code );
    }
}