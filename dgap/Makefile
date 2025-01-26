# See LICENSE.txt for license details.

CXX_FLAGS += -std=c++11 -g -rdynamic -O3 -Wall -g
LDLIBS = -lpmemobj -lpmem -lralloc -lpthread
PAR_FLAG = -fopenmp
RALLOC_PATH = ../ralloc

ifneq (,$(findstring icpc,$(CXX)))
	PAR_FLAG = -openmp
endif

ifneq (,$(findstring sunCC,$(CXX)))
	CXX_FLAGS = -std=c++11 -xO3 -m64 -xtarget=native
	PAR_FLAG = -xopenmp
endif

ifneq ($(SERIAL), 1)
	CXX_FLAGS += $(PAR_FLAG)
endif

KERNELS = bc bfs cc_sv pr
COMMON = benchmark.h reader.h writer.h bitmap.h sliding_queue.h builder.h platform_atomics.h timer.h command_line.h pvector.h util.h 
SUITE = $(KERNELS)

.PHONY: all
all: $(SUITE)

ralloc: src/$(KERNELS).cc src/graph-ralloc.h src/$(COMMON) 
	(cd $(RALLOC_PATH)/test && make libralloc.a)
	$(CXX) -L $(RALLOC_PATH)/test -I$(RALLOC_PATH)/src  $< -o $@ $(CXX_FLAGS) $(LDLIBS)

pmdk: src/%.cc src/$(COMMON) src/graph.h
	$(CXX) $< -o $@ $(CXX_FLAGS) $(LDLIBS)

.PHONY: clean
clean:
	rm -f $(SUITE) test/out/*
