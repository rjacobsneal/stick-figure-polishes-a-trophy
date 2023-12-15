# calls:
CC         = g++
CFLAGS = -c -Wall -O3 -I./ -I/Users/reesejacobsneal/opt/homebrew/opt/jpeg/include
LDFLAGS = -L/Users/reesejacobsneal/opt/homebrew/opt/jpeg/lib -ljpeg
EXECUTABLE = movieMaker

SOURCES    = movieMaker.cpp 
OBJECTS    = $(SOURCES:.cpp=.o)

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o movieMaker
