# See LICENSE.txt for license details.

CXX_FLAGS += -std=c++11 -g -rdynamic -O3 -Wall -g -DNUMA_PMEM -DHASH_MODE
LDLIBS = -lpmemobj -lpmem -lnuma
PAR_FLAG = -fopenmp

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
SUITE = $(KERNELS)

.PHONY: all
all: $(SUITE)

% : src/%.cc src/*.h
	$(CXX) $< -o $@ $(CXX_FLAGS) $(LDLIBS)

.PHONY: clean
clean:
	rm -f $(SUITE) test/out/*
