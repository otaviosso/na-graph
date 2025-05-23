//
// Created by Islam, Abdullah Al Raqibul on 10/1/22.
//

#ifndef GRAPHONE_GAP_PR_H
#define GRAPHONE_GAP_PR_H

#include "gapbs/pvector.h"
#include "gapbs/platform_atomics.h"
#include <numa.h>
#include <math.h>

typedef float ScoreT;
const float kDamp = 0.85;

/***********************************************************************************************/
/**                              PageRank Algorithm                                           **/
/***********************************************************************************************/

/// this code copied from: plaingraph_manager_t<T>::run_pr
//template <class T>
pvector<ScoreT> run_pr(XPGraph* snaph, int max_iters, double epsilon = 0)
{
//  pgraph_t<T>* pgraph = (pgraph_t<T>*)get_plaingraph();
//  snap_t<T>* snaph = create_static_view(pgraph, STALE_MASK|V_CENTRIC);

//  mem_pagerank<dst_id_t>(snaph, 20);
//  delete_static_view(snaph);

  const ScoreT init_score = 1.0f / snaph->get_vcount();
  const ScoreT base_score = (1.0f - kDamp) / snaph->get_vcount();
  pvector<ScoreT> scores(snaph->get_vcount(), init_score);
  pvector<ScoreT> outgoing_contrib(snaph->get_vcount());
  for (int iter=0; iter < max_iters; iter++) {
    double error = 0;
#pragma omp parallel for
    for (vid_t n=0; n < snaph->get_vcount(); n++)
      outgoing_contrib[n] = scores[n] / snaph->get_out_degree(n);
#pragma omp parallel for reduction(+ : error) schedule(dynamic, 64)
    for (vid_t u=0; u < snaph->get_vcount(); u++) {
      ScoreT incoming_total = 0;

      sid_t sid;
      degree_t nebr_count = 0;
      degree_t local_degree = 0;
      vid_t* local_adjlist;
      nebr_count = snaph->get_in_degree(u);
      if (0 == nebr_count) continue;

      local_adjlist = new vid_t[nebr_count];
      local_degree = snaph->get_in_nebrs(u, local_adjlist);
      assert(local_degree == nebr_count);

      // traverse the delta adj list
      for (index_t j = 0; j < local_degree; ++j){
        sid = local_adjlist[j];
        incoming_total += outgoing_contrib[sid];
      }
      delete [] local_adjlist;
      ScoreT old_score = scores[u];
      scores[u] = base_score + kDamp * incoming_total;
      error += fabs(scores[u] - old_score);
    }
  }
  return scores;
}


void bind_thread_to_cpu(int cpu_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu_id, &cpuset);
  
  int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
  if (rc != 0) {
      std::cerr << "Error binding thread to CPU " << cpu_id << std::endl;
  }
}

void bind_cpu_new(int tid) {
  int num_nodes = numa_num_configured_nodes();
  int totalThreads = omp_get_max_threads();
  const int ncores_per_socket = totalThreads / num_nodes / 2; // Adjust based on your system
  const int total_cores = ncores_per_socket * 2; // 24 total cores per NUMA node
  // Select socket by alternating: even tid → socket 0, odd tid → socket 1.

  // Compute the actual CPU ID based on your system's layout.
  if(tid>totalThreads){
    printf("Too many threads\n");
    return;
  }

  int cpu_id=tid/ncores_per_socket;
  int socket = cpu_id%num_nodes;
  //Just working for 2 sockets...
  if(socket==0){
    if(tid>ncores_per_socket){
      bind_thread_to_cpu(tid-(socket+1)*ncores_per_socket);}
    else{
      bind_thread_to_cpu(tid);}

  }  
  else{
      if(cpu_id>num_nodes){
        bind_thread_to_cpu(tid);}
      else{
        bind_thread_to_cpu(tid+socket*ncores_per_socket);}
    }

  
  
/*
  // Debugging printout
  std::cout << "Thread " << tid << " bound to CPU " << cpu_id
            << " (socket " << socket << ", core " << local_index << ")" << std::endl;
*/
}


pvector<ScoreT> run_pr_numa(XPGraph* snaph, int max_iters, double epsilon = 0) {
  const ScoreT init_score = 1.0f / snaph->get_vcount();
  const ScoreT base_score = (1.0f - kDamp) / snaph->get_vcount();
  pvector<ScoreT> scores(snaph->get_vcount(), init_score);
  pvector<ScoreT> outgoing_contrib(snaph->get_vcount());
  vid_t v_count = snaph->get_vcount();
  uint8_t NUM_SOCKETS = numa_num_configured_nodes();
  tid_t ncores_per_socket = omp_get_max_threads() / NUM_SOCKETS / 2; // Adjust based on your system

  for (int iter = 0; iter < max_iters; iter++) {
      double error = 0;

      // First phase: Compute outgoing contributions
      #pragma omp parallel for
      for (vid_t n=0; n < v_count; n++)
        outgoing_contrib[n] = scores[n] / snaph->get_out_degree(n);

      // Second phase: Update scores and compute error
      #pragma omp parallel reduction(+ : error)
      {
          tid_t tid = omp_get_thread_num();
          for (int id = 0; id < NUM_SOCKETS; ++id) {
            if((tid >= ncores_per_socket*id && tid < ncores_per_socket*(id+1)) 
                || (tid >= ncores_per_socket*NUM_SOCKETS + ncores_per_socket*id && tid < ncores_per_socket*NUM_SOCKETS + ncores_per_socket*(id+1))){
              //bind_cpu_new(tid);
              snaph->bind_cpu(tid,id);
              #pragma omp for schedule(dynamic, 4096) nowait
              for (vid_t u = id; u < v_count; u += NUM_SOCKETS) {
                  ScoreT incoming_total = 0;

                  sid_t sid;
                  degree_t nebr_count = 0;
                  degree_t local_degree = 0;
                  vid_t* local_adjlist;
                  nebr_count = snaph->get_in_degree(u);
                  if (0 == nebr_count) continue;

                  local_adjlist = new vid_t[nebr_count];
                  local_degree = snaph->get_in_nebrs(u, local_adjlist);
                  assert(local_degree == nebr_count);

                  // Traverse the delta adj list
                  for (index_t j = 0; j < local_degree; ++j) {
                      sid = local_adjlist[j];
                      incoming_total += outgoing_contrib[sid];
                  }
                  delete[] local_adjlist;

                  ScoreT old_score = scores[u];
                  scores[u] = base_score + kDamp * incoming_total;
                  error += fabs(scores[u] - old_score);
              }
            }
            snaph->cancel_bind_cpu();
              
          }
      }
  }

  return scores;
}



void PrintTopPRScores(XPGraph* snaph, const pvector<ScoreT> &scores) {
  vector<std::pair<vid_t, ScoreT>> score_pairs(snaph->get_vcount());
  for (vid_t n=0; n < snaph->get_vcount(); n++) {
    score_pairs[n] = std::make_pair(n, scores[n]);
  }
  int k = 5;
  vector<std::pair<ScoreT, vid_t>> top_k = TopK(score_pairs, k);
  k = std::min(k, static_cast<int>(top_k.size()));
  for (auto kvp : top_k)
    std::cout << kvp.second << ":" << kvp.first << std::endl;
}

#endif //GRAPHONE_GAP_PR_H
