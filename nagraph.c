#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "pagerank.h"
#include "connected_components.h"
#include "options.h"


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
    // Show the options
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
    // Choose the right benchmark
    if (strcmp(options->application, "pagerank") == 0) {
        printf("Running PageRank...\n");
        run_pagerank(options);
    } else if (strcmp(options->application, "bc") == 0) {
        printf("Running BC...\n");
        // Adicione a lógica para BC aqui
    } else if (strcmp(options->application, "cc") == 0) {
        printf("Running CC...\n");
        run_cc(options);
    } else if (strcmp(options->application, "bfs") == 0) {
        printf("Running BFS...\n");
        // Adicione a lógica para BFS aqui
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
