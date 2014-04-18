#include "fractal.h"
#include <cuda_runtime.h>
#include <cuComplex.h>

using namespace std;


/*struct Complex {
	double Re;
	double Im;

	Complex(double re, double im) {
		this->Re = re;
		this->Im = im;
	}
};*/


__global__ void mandelNum( int *counts, double *data, long *histogram, const int imgWidth, const int imgHeight, const double midX, const double midY,
                           const double scale, const int iterations, const bool julia, const double juliaX, const double juliaY ) {
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = index_y * imgWidth + index_x;
    
    if( index_x < imgWidth && index_y < imgHeight ) {
        int counter = 0;
        double a, b, x, y, ax, ay;
		if (imgWidth > imgHeight) {
			ax = ( ( double )imgWidth ) / imgHeight;
			ay = 1;
		} else {
			ax = 1;
			ay = ( ( double )imgHeight ) / imgWidth;
		}


        a = midX + 2 * ax * scale * ( double )( 2 * index_x - imgWidth ) / imgWidth;
        b = midY + 2 * ay * scale * ( double )( 2 * index_y - imgHeight ) / imgHeight;
        if( julia ) {
            x = juliaX;
            y = juliaY;
        } else {
            x = a;
            y = b;
        }
		

        double asq = a * a, bsq = b * b;
        double atemp;
		double r2 = 2 << 16;

		//calculate mandelnumber
		while( ( asq + bsq < r2 ) && ( counter < iterations ) ) {
			atemp = asq - bsq + x;
			b = a * b;
			b += b + y;
			a = atemp;

			counter++;
			asq = a * a;
			bsq = b * b;
		}
        
		if (counter != 0)
			histogram[counter]++;
		
		counts[i] = counter;
		data[2 * i] = a;
		data[2 * i + 1] = b;
	}
}


__global__ void coloring( int *counts, double *data, long *histogram, GLubyte *array, const int imgWidth, const int imgHeight, const int depth, const int iterations ) {

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = index_y * imgWidth + index_x;

    if( index_x < imgWidth && index_y < imgHeight ) {

		int total = histogram[iterations];
		int j = depth * i;


		double hue = (double)histogram[counts[i]];
			
		if( counts[i] < iterations ) {
			double hue2 = (double)histogram[counts[i] + 1];

			double x = data[2 * i];
			double y = data[2 * i + 1];
			double zn = x * x + y * y;

			//from: COLORING DYNAMICAL SYSTEMSIN THE COMPLEX PLANE
			//http://math.unipa.it/~grim/Jbarrallo.PDF
			double nu = ( log( log( (double)(2 << 16) ) ) - log( (double)(0.5 * log( zn )) ) ) / log( 2. ); //lg(log(b)) - lg(log(sqrt(zn))
			hue += nu * (hue2 - hue);
		} 

		hue /= total;

		//colour scheme
		GLubyte colorArray[] = { 0, 0, 0, 255, 0, 0, 255, 155, 0, 255, 255, 255, 0, 0, 0 };
		int length = 5;

		int n = (int)( hue * ( length - 1 ) );
		double h = hue * ( length - 1 ) - n;

		GLubyte r1 = colorArray[3 * n];
		GLubyte g1 = colorArray[3 * n + 1];
		GLubyte b1 = colorArray[3 * n + 2];
		GLubyte r2 = colorArray[3 * n + 3];
		GLubyte g2 = colorArray[3 * n + 4];
		GLubyte b2 = colorArray[3 * n + 5];
		GLubyte r = r1 * ( 1 - h ) + r2 * h;
		GLubyte g = g1 * ( 1 - h ) + g2 * h;
		GLubyte b = b1 * ( 1 - h ) + b2 * h;

		array[j] =	   (GLubyte)r;
		array[j + 1] = (GLubyte)g;
		array[j + 2] = (GLubyte)b;
	}
}



__global__ void mandelNumExp( GLubyte *array, const int imgWidth, const int imgHeight, const int depth, const double midX, const double midY,
							  const double scale, const int iterations, const bool julia, const double juliaX, const double juliaY ) {

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = index_y * imgWidth + index_x;
    
    if( index_x < imgWidth && index_y < imgHeight ) {
        int counter = 0;
        double a, b, x, y, ax, ay;
		if (imgWidth > imgHeight) {
			ax = ( ( double )imgWidth ) / imgHeight;
			ay = 1;
		} else {
			ax = 1;
			ay = ( ( double )imgHeight ) / imgWidth;
		}


        a = midX + 2 * ax * scale * ( double )( 2 * index_x - imgWidth ) / imgWidth;
        b = midY + 2 * ay * scale * ( double )( 2 * index_y - imgHeight ) / imgHeight;
        if( julia ) {
            x = juliaX;
            y = juliaY;
        } else {
            x = a;
            y = b;
        }

        double asq = a * a, bsq = b * b;
        double atemp, btemp;
		double r = 4;
		double e = 0;

		//double c = -0.7198;
		//double d = 0.9111;
		double c = juliaX;
		double d = juliaY;
		double ctemp, dtemp;

		//c = 0;
		//d = 0;

		//fractal formulas
		//http://www.lifesmith.com/formulas.html

        //calculate mandelnumber
        while( ( asq + bsq < r ) && ( counter < iterations ) ) {
            /*atemp = asq - bsq + x + c;
            b = a * b;
            b += b + y + d;
            a = atemp;*/
			
			//mul(a, b, a, b, &a, &b);
			//add(a, b, x, y, &a, &b);


			//d_function[2 * i](a, b, a, b, &a, &b);
			//d_function[2 * i + 1](a, b, x, y, &a, &b);

			//f_mul(a, b, a, b, &a, &b);
			//f_add(a, b, x, y, &a, &b);

		    mul(a, b, a, b, &a, &b);
			add(a, b, x, y, &a, &b);
			add(a, b, c, d, &a, &b);

			/*mul(a, b, a, b, &a, &b);
			mul(a, b, x, y, &a, &b);
			add(a, b, c, d, &a, &b);*/

			/*atemp = a * a * a - 3 * a * b * b + a * c - b * d + x;
			b = 3 * a * a * b - b * b * b + a * d + b * c + y;
			a = atemp;*/
			
			/*mul(a, b, c, d, &ctemp, &dtemp);
			//pow(a, b, 3, &a, &b);
			mul(a, b, a, b, &atemp, &btemp);
			mul(a, b, atemp, btemp, &a, &b);
			add(a, b, ctemp, dtemp, &a, &b);
			add(a, b, x, y, &a, &b);*/

			
			counter++;
			e += expf( - ( asq + bsq ) ); //do not use sqrt as it does not add much

            asq = a * a;
            bsq = b * b;
        }
        
		float hue;
		
		if (counter == iterations) {
			hue = 1;
		} else {
			hue = ( 0.025f * e - (int)(0.025f * e) );
		}


		GLubyte R, G, B;

		//colour scheme
		GLubyte colorArray[] = { 0, 0, 0, 255, 0, 0, 255, 155, 0, 255, 255, 255, 0, 0, 0 };
		int length = 5;

		int n = (int)( hue * ( length - 1 ) );
		float h = hue * ( length - 1 ) - n;

		GLubyte r1 = colorArray[3 * n];
		GLubyte g1 = colorArray[3 * n + 1];
		GLubyte b1 = colorArray[3 * n + 2];
		GLubyte r2 = colorArray[3 * n + 3];
		GLubyte g2 = colorArray[3 * n + 4];
		GLubyte b2 = colorArray[3 * n + 5];
		R = r1 * ( 1 - h ) + r2 * h;
		G = g1 * ( 1 - h ) + g2 * h;
		B = b1 * ( 1 - h ) + b2 * h;


		/*double H = 1 - hue;
		double L = hue;
		double S = 1;

		double C = (1 - abs( 2 * L - 1 ) ) * S;
		double Hprime = 6 * H; //H should be in [0,360), and H' in [0,6), but H is in [0, 1), so we do this instead.
        double X = C * (double)(1 - abs( Hprime - 2 * (int)( Hprime / 2 ) - 1 ) ); //C * (1 - |H' mod 2 - 1|)
        double m = L - C / 2;

        GLubyte bC, bX, b0;
        bC = (GLubyte)( ( C + m ) * 255 );
        bX = (GLubyte)( ( X + m ) * 255 );
        b0 = (GLubyte)( ( 0 + m ) * 255 );

		if		(Hprime < 1) { R = bC; G = bX; B = b0; }
		else if (Hprime < 2) { R = bX; G = bC; B = b0; }
		else if (Hprime < 3) { R = b0; G = bC; B = bX; }
		else if (Hprime < 4) { R = b0; G = bX; B = bC; }
		else if (Hprime < 5) { R = bX; G = b0; B = bX; }
		else if (Hprime < 6) { R = bC; G = b0; B = bC; }
		else			  { R = 0;  G = 0;  B = 0; }*/


		int j = depth * i;
		array[j] = R;
		array[j + 1] = G;
		array[j + 2] = B;
	}
}


__global__ void partialSum( long *input, const int length ) {

	int id = threadIdx.x + blockDim.x * threadIdx.y + 
			(blockIdx.x * blockDim.x * blockDim.y) + 
			(blockIdx.y * blockDim.x * blockDim.y * gridDim.x);

	if (id == 0) {
		for (int i = 1; i < length; i++) {
			input[i] += input[i - 1];
		}
	}
}


__device__ void mul( double a, double b, double c, double d, double *x, double *y ) {
	double re = a * c - b * d;
	double im = a * d + b * c;
	*x = re;
	*y = im;
}

__device__ void add( double a, double b, double c, double d, double *x, double *y ) {
	double re = a + c;
	double im = b + d;
	*x = re;
	*y = im;
}

__device__ void pow( double a, double b, double n, double *x, double *y ) {
	double re = 1;
	double im = 0;

	for (int i = 0; i < n; ++i) {
		mul(re, im, a, b, &re, &im);
	}

	*x = re;
	*y = im;
}

__device__ void exp( double a, double b, double *x, double *y ) {
	double ea = exp(a);
	double re = ea * cos(b);
	double im = ea * sin(b);

	*x = re;
	*y = im;
}


//__device__ func f_mul = mul;
//__device__ func f_add = add;



void CalcFractal( GLubyte *devArray, int *counts, double *data, long *histogram, double dPosX, double dPosY, double dScale, int iWidth, int iHeight, 
				  int iDepth, int iIterations, bool bIsJulia, double dJuliaX, double dJuliaY) {
	dim3 blockSize;
    blockSize.x = 8;
    blockSize.y = 8;

    dim3 gridSize;
    gridSize.x = iWidth / blockSize.x;
    gridSize.y = iHeight / blockSize.y;
    
    /*int n = 2;
	func *h_function;
	func *d_function;
	h_function = (func*)malloc(n * iWidth * iHeight * sizeof(func));
	cudaMalloc((void**) &d_function, n * iWidth * iHeight * sizeof(func));

	for (int i = 0; i < iWidth * iHeight; ++i) {
		cudaMemcpyToSymbol(&h_function[2 * i], f_mul, sizeof(func));
		cudaMemcpyToSymbol(&h_function[2 * i + 1], f_add, sizeof(func));
	}

	cudaMemcpy(d_function, h_function, n * iWidth * iHeight * sizeof(func), cudaMemcpyHostToDevice);
	*/

	//switch( type ) {
	//	case FRACTAL_RENDERTYPE_HISTOGRAM:	
			//set histogram to 0 for all values
			/*cudaMemset( histogram, 0, (iIterations + 1) * sizeof(long) );  
    
			mandelNum <<< gridSize, blockSize >>> ( counts, data, histogram, iWidth, iHeight, dPosX, dPosY, dScale, iIterations, bIsJulia, dJuliaX, dJuliaY );
			partialSum <<< gridSize, blockSize >>> ( histogram, iIterations + 1 );
			coloring <<< gridSize, blockSize >>> ( counts, data, histogram, devArray, iWidth, iHeight, iDepth, iIterations );*/
	
	//		break;
	//	case FRACTAL_RENDERTYPE_EXPONENTIAL:
			//mandelNumExp <<< gridSize, blockSize >>> ( devArray, iWidth,iHeight, iDepth, dPosX, dPosY, dScale, 
			//										   iIterations, bIsJulia, dJuliaX, dJuliaY, d_function );

	mandelNumExp <<< gridSize, blockSize >>> ( devArray, iWidth,iHeight, iDepth, dPosX, dPosY, dScale, 
													   iIterations, bIsJulia, dJuliaX, dJuliaY );
	
	//		break;
	//}

	//free(h_function);
	//cudaFree(d_function);

    cudaDeviceSynchronize();
}