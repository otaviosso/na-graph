#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    char *allocator;
    char *dataset;
    int threads;
    char *application;
    char *numamode;
    char *backend;
} Options;

void print_help() {
    printf("Usage: na-graph --allocator <value> --dataset <value> --threads <value>\n");
    printf("             --application <value> --numamode <value> --backend <value>\n");
}

int parse_options(int argc, char *argv[], Options *options) {
    int opt;
    while ((opt = getopt(argc, argv, "a:d:t:A:n:b:")) != -1) {
        switch (opt) {
            case 'a':
                options->allocator = optarg;
                break;
            case 'd':
                options->dataset = optarg;
                break;
            case 't':
                options->threads = atoi(optarg);
                break;
            case 'A':
                options->application = optarg;
                break;
            case 'n':
                options->numamode = optarg;
                break;
            case 'b':
                options->backend = optarg;
                break;
            default:
                print_help();
                return 1;
        }
    }

    if (optind < argc) {
        fprintf(stderr, "Unknown arguments: ");
        while (optind < argc)
            fprintf(stderr, "%s ", argv[optind++]);
        fprintf(stderr, "\n");
        print_help();
        return 1;
    }

    return 0;
}

void run_benchmark(Options *options) {
    // Aqui você implementará a lógica para executar o benchmark
    printf("Running benchmark with options:\n");
    printf("Allocator: %s\n", options->allocator);
    printf("Dataset: %s\n", options->dataset);
    printf("Threads: %d\n", options->threads);
    printf("Application: %s\n", options->application);
    if (options->numamode != NULL)
        printf("NUMAMode: %s\n", options->numamode);
    else
        printf("NUMAMode: Not specified\n");
    printf("Backend: %s\n", options->backend);

    // Aqui você adicionará a lógica para executar o benchmark específico
    // usando os frameworks e benchmarks especificados.
    if (strcmp(options->application, "pagerank") == 0) {
        run_pagerank(options);
    } else if (strcmp(options->application, "bc") == 0) {
        // Adicione a lógica para BC aqui
        printf("Running BC...\n");
    } else if (strcmp(options->application, "cc") == 0) {
        // Adicione a lógica para CC aqui
        printf("Running CC...\n");
    } else if (strcmp(options->application, "bfs") == 0) {
        // Adicione a lógica para BFS aqui
        printf("Running BFS...\n");
    } else {
        fprintf(stderr, "Unsupported application: %s\n", options->application);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    Options options = {0};

    if (parse_options(argc, argv, &options) != 0)
        return EXIT_FAILURE;

    if (options.allocator == NULL || options.dataset == NULL ||
        options.threads <= 0 || options.application == NULL || options.backend == NULL) {
        fprintf(stderr, "Missing required arguments.\n");
        print_help();
        return EXIT_FAILURE;
    }

    run_benchmark(&options);

    return EXIT_SUCCESS;
}
