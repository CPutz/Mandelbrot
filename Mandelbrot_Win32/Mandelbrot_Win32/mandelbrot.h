#include <GL/glut.h>
#include <boost/thread.hpp>
#include "fractal.h"

class Mandelbrot {
private:
	double dPosX;
	double dPosY;
	int iWidth;
	int iHeight;
	int iDepth;
	int iSize;
	double dScale;
	
	GLuint tex;
	GLuint buffer;
	GLubyte *devArray;
	int *devCounts;
	double *devData;
	
	long *devHistogram;
	int iOldIterations;
	
	bool bIsJulia;
	double dJuliaX;
	double dJuliaY;

	bool bIsFinished;

	//void calc( double dPosX, double dPosY, double dScale, int iWidth, int iHeight );
	void createBuffer( GLuint *b, int size );
	void createTexture( GLuint *t, int w, int h );
	void createHistogram( int iterations );
	void calculate( int iterations );
	
public:
	Mandelbrot(int iWidth, int iHeight);
	~Mandelbrot();
	void Initialize();
	void Update(int iterations);
	void Resize(int iWidth, int iHeight);
	void setPosition(double x, double y);
	void setScale(double dScale);
	void setJulia(double x, double y);
	void setIsJulia(bool val);
	bool getIsJulia();
	bool isFinished();
	GLuint getTexture();
	GLuint getBuffer();
};