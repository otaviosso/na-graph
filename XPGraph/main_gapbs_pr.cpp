// main_gapbs_pr.cpp
#include <iostream>
#include "apps/graph_benchmarks.hpp"

int main(int argc, const char ** argv) {
    XPGraph *xpgraph = new XPGraph(argc, argv);
    xpgraph->import_graph_by_config();
    uint8_t count = xpgraph->get_query_count();
    if (count == 0) return 0;

    while (count--) {
        test_gapbs_pr_numa(xpgraph);
        std::cout << std::endl;
    }
    
    xpgraph->save_gragh();
    delete xpgraph;
    return 0;
}
