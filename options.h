#ifndef OPTIONS_H
#define OPTIONS_H

// Definição da estrutura Options
typedef struct {
    char *allocator;
    char *dataset;
    int threads;
    char *application;
    char *numamode;
    char *backend;
} Options;

#endif // OPTIONS_H
