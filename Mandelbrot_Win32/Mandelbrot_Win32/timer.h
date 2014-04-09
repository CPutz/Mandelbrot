#include <time.h>

enum TimeFormat { TIME_FORMAT_MILLI_SEC, TIME_FORMAT_SEC };

class Timer {
    private:
		clock_t tStart;
		clock_t tStop;
        bool isRunning;
        bool wasStarted;
		
    public:
        Timer( bool start );
        void start();
		void stop();
        int stop( TimeFormat t );
        int getTime( TimeFormat t );
};