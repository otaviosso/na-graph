# Compilador e flags
CC = gcc
CXX = g++
CFLAGS = -Wall -O2 -fopenmp
CXXFLAGS = -std=c++11 -g -rdynamic -O3 -Wall -fopenmp -D_GNU_SOURCE -Dnagraph
LDFLAGS = -lstdc++ -lpthread -lpmemobj -lpmem -fopenmp

# Nome do executável final
TARGET = nagraph

# Arquivos fonte
SRC_C = nagraph.c pagerank.c connected_components.c
SRC_CPP = dgap/src/pr.cc dgap/src/cc_sv.cc

# Arquivos objeto
OBJ_C = $(SRC_C:.c=.o)
OBJ_CPP = $(SRC_CPP:.cc=.o)
OBJ = $(OBJ_C) $(OBJ_CPP)

# Regra padrão
all: $(TARGET)

# Regra para compilar o executável
$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET) $(LDFLAGS)

# Regra para compilar arquivos .c
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Regra para compilar arquivos .cc
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Limpar arquivos gerados
clean:
	rm -f $(OBJ) $(TARGET)

# Phony targets
.PHONY: all clean