#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "connected_components.h"
#include "dgap/src/cc_sv.h" // Path to DGAP CC




void run_cc(Options *options) {
    if (strcmp(options->backend, "DGAP") == 0) {
        printf("Running DGAP PageRank...\n");
        // Arguments for DGAP
        const char* base_graph = "/caminho/para/base.el"; // Change to the right file
        const char* dynamic_graph = "/caminho/para/dynamic.el"; // Change to the right file
        const char* pmem_path = "/caminho/para/graph.pmem"; // Change to the right file
        const char* source_vertex = "1"; // Initial vertice, normally 1
        const char* num_trials = "5"; // Repeat number

        // Making the arguments array
        char* argv[] = {
            "dgap_pagerank", // Program name
            "-B", strdup(base_graph),
            "-D", strdup(dynamic_graph),
            "-f", strdup(pmem_path),
            "-r", strdup(source_vertex),
            "-n", strdup(num_trials),
            "-a", // -a
            NULL // Array end
        };
        run_dgap_cc(12, argv); //Number of variables and the array
    } else if (strcmp(options->backend, "CSR") == 0) {
        // Add CSR
        printf("Running CSR PageRank...\n");
    } else if (strcmp(options->backend, "XPGraph") == 0) {
        // Add XPGraph
        printf("Running XPGraph PageRank...\n");
    } else if (strcmp(options->backend, "GraphOne") == 0) {
        // Add GraphOne
        printf("Running GraphOne PageRank...\n");
    } else {
        fprintf(stderr, "Unsupported backend: %s\n", options->backend);
        exit(EXIT_FAILURE);
    }
}

