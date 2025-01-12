// pagerank.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pagerank.h"
#include "dgap/src/pr.h" // Inclua o cabeçalho do DGAP




void run_pagerank(Options *options) {
    if (strcmp(options->backend, "DGAP") == 0) {
        printf("Running DGAP PageRank...\n");
        const char* base_graph = "/caminho/para/base.el"; // Substitua pelo caminho correto
        const char* dynamic_graph = "/caminho/para/dynamic.el"; // Substitua pelo caminho correto
        const char* pmem_path = "/caminho/para/graph.pmem"; // Substitua pelo caminho correto
        const char* source_vertex = "1"; // Vértice inicial
        const char* num_trials = "5"; // Número de tentativas
        const char* analysis_flag = "-a"; // Flag para análise

        // Simule os argumentos que seriam passados para o DGAP
        char* argv[] = {
            "dgap_pagerank",
            "-B", strdup(base_graph), // Carregar o grafo base
            "-D", strdup(dynamic_graph), // Inserir o grafo dinâmico
            "-f", strdup(pmem_path), // Armazenar o grafo em pmem
            "-r", strdup(source_vertex), // Vértice inicial
            "-n", strdup(num_trials), // Número de tentativas
            analysis_flag, // Flag para análise
            NULL
        };
        run_dgap_pagerank(8, argv);
    } else if (strcmp(options->backend, "CSR") == 0) {
        // Adicione a lógica para CSR aqui
        printf("Running CSR PageRank...\n");
    } else if (strcmp(options->backend, "XPGraph") == 0) {
        // Adicione a lógica para XPGraph aqui
        printf("Running XPGraph PageRank...\n");
    } else if (strcmp(options->backend, "GraphOne") == 0) {
        // Adicione a lógica para GraphOne aqui
        printf("Running GraphOne PageRank...\n");
    } else {
        fprintf(stderr, "Unsupported backend: %s\n", options->backend);
        exit(EXIT_FAILURE);
    }
}

