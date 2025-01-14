#ifndef OPTIONS_H
#define OPTIONS_H

// Definition of Options
typedef struct {
    char *allocator;
    char *dataset;
    int threads;
    char *application;
    char *numamode;
    char *backend;
} Options;

#endif // OPTIONS_H
