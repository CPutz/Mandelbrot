#define GLEW_STATIC
#define CUDA_STATIC

#include <iostream>
#include <sstream>
#include <math.h>
#include <stdlib.h>

#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "timer.h"
#include "mandelbrot.h"
#include "cuda_helper.h"

using namespace std;


#define BUTTON_UP				 0
#define BUTTON_DOWN_UNREGISTERED 1
#define BUTTON_DOWN_REGISTERED   2

__event
unsigned int windowWidth = 512;
unsigned int windowHeight = 512;

GLuint buffer;
GLuint tex;
GLubyte *d_array;
Mandelbrot *fractal;

int depth = 4;
int imgWidth = windowWidth;
int imgHeight = windowHeight;
int imgSize = imgWidth * imgHeight * depth * sizeof( GLubyte );
float iSamplingFactor = 1;

int lastTime, currentTime;
bool rightDown, leftDown;
double zoom = 0.0f;
double zoomspeed = 1.0f;
double scale = 1.0f;
double xMid = 0;
double yMid = 0;
double xPos, yPos;
int iterations = 128;
bool changePos = false;

unsigned char keyState[255];
unsigned char specialKeyState[255];
int keyModifiers;

int renderTime;
int processTime;
Timer *timer;

int frame = 0;
Timer *fpsTimer;
int fps = 30;
bool iterationChanged = false;


void drawText( int x, int y, string s ) {
    glRasterPos2i( x, y );
    for( size_t i = 0; i < s.size(); ++i ) {
        glutBitmapCharacter( GLUT_BITMAP_HELVETICA_18, s[i] );
    }
}



void setOrthographicProjection() {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, windowWidth, 0, windowHeight, -1, 1);
    glMatrixMode(GL_MODELVIEW);
}



void resetPerspectiveProjection() {
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}



void displayFunc() {
    timer->start();
	setOrthographicProjection();

    glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );
    glClear( GL_COLOR_BUFFER_BIT );

    glEnable( GL_TEXTURE_2D );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, buffer );
    glBindTexture( GL_TEXTURE_2D, tex );
    glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, imgWidth, imgHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0 );


	double w = 2 * scale;
	double h = 2 * scale;
	double x1 = (xMid - w / 2 + 1) / 2;
	double x2 = (xMid + w / 2 + 1) / 2;
	double y1 = (yMid - h / 2 + 1) / 2;
	double y2 = (yMid + h / 2 + 1) / 2;


    glBegin( GL_QUADS );

	glColor3f( 1.0f, 1.0f, 1.0f );
    glTexCoord2d( x1, y1 );
    glVertex2d( 0, 0 );
    glTexCoord2d( x2, y1 );
    glVertex2d( windowWidth, 0 );
    glTexCoord2d( x2, y2 );
    glVertex2d( windowWidth, windowHeight );
    glTexCoord2d( x1, y2 );
    glVertex2d( 0, windowHeight );

    glEnd();
    

    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
    glBindTexture( GL_TEXTURE_2D, 0 );
    glDisable( GL_TEXTURE_2D );

    //draw text
    glColor3f( 1.0f, 1.0f, 1.0f );
    drawText( 0, 0,	 "RenderTime: "	  + to_string( (long long)renderTime ) + " msec" );
    drawText( 0, 18, "ProcessTime: "  + to_string( (long long)processTime ) + " msec" );
    drawText( 0, 36, "zoom: "		  + to_string( (double long)1 / scale ) + "x" );
    drawText( 0, 54, "zoomspeed: "	  + to_string( (double long)pow ( 2, zoomspeed ) ) );
    drawText( 0, 72, "iterations: "	  + to_string( (long long)iterations ) );
	drawText( 0, 90, "samplefactor: " + to_string( (double long)iSamplingFactor ) );

    drawText( 0, windowHeight - 18, "fps: " + to_string( (long long)fps ) );
	drawText( 0, windowHeight - 36, "pos: " + to_string( (double long)xPos ) + ", " + to_string( (double long)yPos ) );
    
    resetPerspectiveProjection();

    glutSwapBuffers();

    renderTime = timer->stop( TIME_FORMAT_MILLI_SEC );

	frame++;
	if (fpsTimer->getTime( TIME_FORMAT_MILLI_SEC ) > 1000) {
		fps = frame * 1000 / fpsTimer->getTime( TIME_FORMAT_MILLI_SEC );
		fpsTimer->stop();
		fpsTimer->start();
		frame = 0;
		iterationChanged = false;
	}
}


void reshapeFunc( int w, int h ) {
	glViewport(0, 0, w, h);

    windowWidth = w;
    windowHeight = h;

    imgWidth = windowWidth * iSamplingFactor;
    imgHeight = windowHeight * iSamplingFactor;
    imgSize = windowWidth * windowHeight * depth * sizeof( GLubyte );

	fractal->Resize(imgWidth, imgHeight);

	tex = fractal->getTexture();
    buffer = fractal->getBuffer();

	glutPostRedisplay();
}


void handleKeyboard( int timeElapsed ) {
	if ( specialKeyState[GLUT_KEY_PAGE_UP] == BUTTON_DOWN_UNREGISTERED) {
		if (keyModifiers == GLUT_ACTIVE_CTRL) {
			iSamplingFactor = iSamplingFactor + 0.1;
			reshapeFunc( windowWidth, windowHeight );
		} else {
			iterations = iterations << 1;
		}
		specialKeyState[GLUT_KEY_PAGE_UP] = BUTTON_DOWN_REGISTERED;
	}
	if ( specialKeyState[GLUT_KEY_PAGE_DOWN] == BUTTON_DOWN_UNREGISTERED) {
		if (keyModifiers == GLUT_ACTIVE_CTRL) {
			if (iSamplingFactor > 0.1) {
				iSamplingFactor = iSamplingFactor - 0.1;
				reshapeFunc( windowWidth, windowHeight );
			}
		} else {
			if (iterations > 2)
				iterations = iterations >> 1;
		}
		specialKeyState[GLUT_KEY_PAGE_DOWN] = BUTTON_DOWN_REGISTERED;
	}

    
	if ( keyState['j'] == BUTTON_DOWN_UNREGISTERED ) {
        fractal->setIsJulia( !fractal->getIsJulia() );
		keyState['j'] = BUTTON_DOWN_REGISTERED;
	}
	if ( keyState['a'] == BUTTON_DOWN_UNREGISTERED ) {
		//if( fractal->getIsJulia() )
			changePos = !changePos;
		keyState['a'] = BUTTON_DOWN_REGISTERED;
	}
	if ( keyState['p'] == BUTTON_DOWN_UNREGISTERED ) {
		/*glEnable( GL_TEXTURE_2D );
        char *imgData = new char[windowWidth * windowHeight * depth];
        glGetTexImage( GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, imgData );

        cout << "printing screen to: screen.bmp" << endl;
        //FILE file = fOpen(
        delete( imgData );*/
		keyState['p'] = BUTTON_DOWN_REGISTERED;
	}

	if ( keyState['+'] == BUTTON_DOWN_UNREGISTERED ) {
		zoomspeed += ( (double)timeElapsed / 1000 );
    }
	if ( keyState['-'] == BUTTON_DOWN_UNREGISTERED ) {
		zoomspeed -= ( (double)timeElapsed / 1000 );
    }
}



void idleFunc() {
    lastTime = currentTime;
    currentTime = glutGet( GLUT_ELAPSED_TIME );

    int delta = currentTime - lastTime;

    if( ( leftDown || rightDown ) && !( /*fractal->getIsJulia() &&*/ changePos ) ) {
        xMid += ( xPos - xMid ) * delta * 0.001 * zoomspeed;
        yMid += ( yPos - yMid ) * delta * 0.001 * zoomspeed;
        if( leftDown ) {
            zoom -= ( double ) delta * 0.001 * zoomspeed;
        } else {
            zoom += ( double ) delta * 0.001 * zoomspeed;
        }

		scale = pow( 2, zoom );

		fractal->setPosition( xMid, yMid );
		fractal->setScale( scale );
    }

    
	handleKeyboard( delta );
    
	timer->start();
	fractal->Update( iterations );
	processTime = timer->stop( TIME_FORMAT_MILLI_SEC );
	   
	/*if ( !iterationChanged ) {
		if (fps > 60 && iterations < (2 << 9)) {
			iterations = iterations << 1;
			iterationChanged = true;
		} else if (fps < 25 && iterations > 2) {
			iterations = iterations >> 1;
			iterationChanged = true;
		}
	}*/

    glutPostRedisplay();
}



void updateMousePos( int x, int y ) {

	xPos = xMid + scale * 4 * ( double )( x - ( double ) windowWidth / 2 ) / windowWidth;
    yPos = yMid + scale * 4 * ( double )( ( double ) windowHeight / 2 - y ) / windowHeight;
}



void mouseFunc( int button, int buttonState, int x, int y ) {
    updateMousePos( x, y );

    if( buttonState == GLUT_DOWN ) {
        if( button == 0 ) {
            leftDown = true;
        } else if( button == 2 ) {
            rightDown = true;
        }
        currentTime = glutGet( GLUT_ELAPSED_TIME );
    } else if( buttonState == GLUT_UP ) {
        if( button == 0 ) {
            leftDown = false;
        } else if( button == 2 ) {
            rightDown = false;
        }
    }
}



void passiveMouseFunc( int x, int y ) {
	updateMousePos( x, y );
}



void motionFunc( int x, int y ) {
	updateMousePos( x, y );

    if( /*fractal->getIsJulia() &&*/ changePos ) {
        fractal->setJulia( xPos, yPos );
    } else if( leftDown || rightDown ) {
        
    }
}



void specialFunc( int key, int x, int y ) {
	if ( specialKeyState[key] != BUTTON_DOWN_REGISTERED )
		specialKeyState[key] = BUTTON_DOWN_UNREGISTERED;
	keyModifiers = glutGetModifiers();
}

void keyboardFunc( unsigned char key, int x, int y ) {
	if ( keyState[key] != BUTTON_DOWN_REGISTERED )
		keyState[key] = BUTTON_DOWN_UNREGISTERED;
	keyModifiers = glutGetModifiers();
}

void specialUpFunc( int key, int x, int y ) {
	specialKeyState[key] = BUTTON_UP;
	keyModifiers = glutGetModifiers();
}

void keyboardUpFunc( unsigned char key, int x, int y ) {
   keyState[key] = BUTTON_UP;
   keyModifiers = glutGetModifiers();
}



void setCudaDevice() {
    int numDevices;
    int max = 0, maxDevice = 0;
    cudaGetDeviceCount( &numDevices );
    for( int i = 0; i < numDevices; ++i ) {
        cudaDeviceProp properties;
        checkCudaErrors( cudaGetDeviceProperties( &properties, i ), __LINE__, false );
        if( max < properties.multiProcessorCount ) {
            max = properties.multiProcessorCount;
            maxDevice = i;
        }
    }

    if( numDevices > 0 ) {
        checkCudaErrors( cudaSetDevice( maxDevice ), __LINE__, true );

        cudaDeviceProp devProp;
        checkCudaErrors( cudaGetDeviceProperties( &devProp, maxDevice ), __LINE__, false );
        cout << "Using device: " << devProp.name << endl;
    } else {
        cout << "No CUDA device found." << endl;
        exit( -1 );
    }
}



void cleanup() {
	delete fractal;
	delete timer;
	delete fpsTimer;
}



int main( int argc, char **argv ) {

    //Initiate glut
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( windowWidth, windowHeight );
    glutInitWindowPosition( 0, 0 );
    glutCreateWindow( "Mandelbrot" );

	//set glut update functions
    glutDisplayFunc( displayFunc );
    glutIdleFunc( idleFunc );
    glutMouseFunc( mouseFunc );
	glutPassiveMotionFunc( passiveMouseFunc );
    glutMotionFunc( motionFunc );
    glutKeyboardFunc( keyboardFunc );
	glutSpecialFunc( specialFunc );
	glutKeyboardUpFunc( keyboardUpFunc );
	glutSpecialUpFunc( specialUpFunc );
    glutReshapeFunc( reshapeFunc );

    //set best CUDA device
    setCudaDevice();

    //Create fractal
    fractal = new Mandelbrot( imgWidth * iSamplingFactor, imgHeight * iSamplingFactor );
    fractal->Initialize();
    //fractal->setJulia(-0.92, -0.29);
	fractal->setJulia(0,0);
    tex = fractal->getTexture();
    buffer = fractal->getBuffer();
	
	timer = new Timer( false );
	fpsTimer = new Timer( true );

    atexit( cleanup );

    glutMainLoop();

    return 0;
}
