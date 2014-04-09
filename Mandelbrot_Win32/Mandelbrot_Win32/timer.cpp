#include "timer.h"
#include <GL/glut.h>

Timer::Timer( bool start ) {
    if( start )
        Timer::start();
    else {
        this->isRunning = false;
        this->wasStarted = false;
    }
}

void Timer::start() {
	this->tStart = glutGet( GLUT_ELAPSED_TIME );
    this->isRunning = true;
    this->wasStarted = true;
}

int Timer::getTime( TimeFormat t ) {
	if ( this->isRunning ) {
		this->tStop = glutGet( GLUT_ELAPSED_TIME );
	}
	
	if ( this->wasStarted ) {
        switch( t ) {
            case TIME_FORMAT_SEC:
				return ( this->tStop - this->tStart ) / 1000;
            case TIME_FORMAT_MILLI_SEC:
                return this->tStop - this->tStart;
            default:
                return -1;
        }
    } else {
        return -1;
    }
}

void Timer::stop() {
	if( this->isRunning ) {
		this->tStop = glutGet( GLUT_ELAPSED_TIME );
    }
    this->isRunning = false;
}

int Timer::stop( TimeFormat t ) {
	stop();
    return getTime( t );
}