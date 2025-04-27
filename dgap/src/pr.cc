// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "pr.h"
#include <omp.h>
#include <string.h>
/*
GAP Benchmark Suite
Kernel: PageRank (PR)
Author: Scott Beamer

Will return pagerank scores for all vertices once total change < epsilon

This PR implementation uses the traditional iterative approach. This is done
to ease comparisons to other implementations (often use same algorithm), but
it is not necesarily the fastest way to implement it. It does perform the
updates in the pull direction to remove the need for atomics.
*/


using namespace std;

typedef float ScoreT;
const float kDamp = 0.85;
void PrintTopScores(const WGraph &g, const ScoreT *scores);
void bind_current_thread_to_cpu_list(const std::vector<int> &cpus);

ScoreT *PageRankPull(const WGraph &g, int max_iters, double epsilon = 0) {
  const int64_t N          = g.num_nodes();
  const ScoreT  init_score = 1.0f / N;
  const ScoreT  base_score = (1.0f - kDamp) / N;
  ScoreT       *scores     = (ScoreT*)malloc(sizeof(ScoreT)*N);
  ScoreT       *outgoing   = (ScoreT*)malloc(sizeof(ScoreT)*N);
  int64_t       vertices0  = N/2;

  // inicializa
  for (int64_t i = 0; i < N; ++i)
    scores[i] = init_score;

  double error = 0.0;

  #pragma omp parallel
  {
    // Bind uma única vez por thread
    int tid        = omp_get_thread_num();
    int numThreads = omp_get_num_threads();
    int n0         = numThreads/2;

    static const std::vector<int> node0_cpus = {
       0,1,2,3,4,5,6,7,8,9,10,11,
      24,25,26,27,28,29,30,31,32,33,34,35
    };
    static const std::vector<int> node1_cpus = {
      12,13,14,15,16,17,18,19,20,21,22,23,
      36,37,38,39,40,41,42,43,44,45,46,47
    };
    int64_t start, end;
    if (tid < n0){
      bind_current_thread_to_cpu_list(node0_cpus);
      int64_t chunk = (vertices0 + n0 - 1) / n0;
      start = tid * chunk;
      end   = std::min<int64_t>(vertices0, start + chunk);
    }
    else{
      bind_current_thread_to_cpu_list(node1_cpus);
      int   tid1 = tid - n0;
      int64_t rem   = N - vertices0;
      int64_t chunk = (rem + (numThreads-n0) - 1) / (numThreads-n0);
      start = vertices0 + tid1 * chunk;
      end   = std::min<int64_t>(N, start + chunk);
    }
    // loop -> reusa start/end sem recalcular
    for (int iter = 0; iter < max_iters; ++iter) {
      ScoreT local_err = 0;

      for (int64_t u = start; u < end; ++u) //Utiliza o intervalo criado
        outgoing[u] = scores[u] / g.out_degree(u);

      // Pagerank em si, também utiliza o itervalo criado
      for (int64_t u = start; u < end; ++u) {
        ScoreT sum = 0;
        for (auto v : g.in_neigh(u))
          sum += outgoing[v];
        ScoreT old = scores[u];
        scores[u] = base_score + kDamp * sum;
        local_err += fabs(scores[u] - old);
      }

      #pragma omp atomic
      error += local_err;
    }
  } // fim do parallel

  return scores;
}


void bind_current_thread_to_cpu_list(const std::vector<int> &cpus) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (int cpu : cpus) {
      CPU_SET(cpu, &cpuset);
  }
  pthread_t tid = pthread_self();
  pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
}


void PrintTopScores(const WGraph &g, const ScoreT *scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n=0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  k = min(k, static_cast<int>(top_k.size()));
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}


// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
bool PRVerifier(const WGraph &g, const ScoreT *scores,
                        double target_error) {
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> incomming_sums(g.num_nodes(), 0);
  double error = 0;
  for (NodeID u : g.vertices()) {
    ScoreT outgoing_contrib = scores[u] / g.out_degree(u);
    for (NodeID v : g.out_neigh(u))
      incomming_sums[v] += outgoing_contrib;
  }
  for (NodeID n : g.vertices()) {
    error += fabs(base_score + kDamp * incomming_sums[n] - scores[n]);
    incomming_sums[n] = 0;
  }
  PrintTime("Total Error", error);
  return error < target_error;
}

#ifdef nagraph
int run_dgap_pagerank(int argc, char* argv[]) {
  CLPageRank cli(argc, argv, "pagerank", 1e-4, 20);
  if (!cli.ParseArgs())
    return -1;
  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();

//  g.print_pma_meta();
  auto PRBound = [&cli] (const WGraph &g) {
    return PageRankPull(g, cli.max_iters(), cli.tolerance());
  };
  auto VerifierBound = [&cli] (const WGraph &g, const ScoreT *scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
  BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
  return 0;
}

#else
int main(int argc, char* argv[]) {
  CLPageRank cli(argc, argv, "pagerank", 1e-4, 20);
  if (!cli.ParseArgs())
    return -1;
  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  printf("deu certo criar grafo\n");
//  g.print_pma_meta();
  auto PRBound = [&cli] (const WGraph &g) {
    return PageRankPull(g, cli.max_iters(), cli.tolerance());
  };
  auto VerifierBound = [&cli] (const WGraph &g, const ScoreT *scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
  BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
  return 0;
}
#endif
