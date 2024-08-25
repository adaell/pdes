
CXX=/usr/bin/g++
PYTHON=/usr/bin/python3

all:
	${CXX} -march=native -O3 -fopenmp nbody.cpp -o nbody

debug:
	${CXX} -p -pg -g nbody.cpp -o nbody

data:
	make all
	./nbody > nbody.dat

image:
	${PYTHON} parse_nbody_output.py
	${PYTHON} ../make_gif.py

clean:
	rm -f nbody
	rm -f nbody.dat
	rm -f images/*
	rm -f animation.gif

nbody:
	@make -s clean
	@echo compiling
	@make -s all
	@echo running the simulation
	@make -s data
	@echo making the images
	@make -s image