// main_gapbs_bfs.cpp
#include <iostream>
#include "apps/graph_benchmarks.hpp"

int main(int argc, const char ** argv) {
    XPGraph *xpgraph = new XPGraph(argc, argv);
    xpgraph->import_graph_by_config();

    test_gapbs_bfs(xpgraph);

    xpgraph->save_gragh();
    delete xpgraph;
    return 0;
}
