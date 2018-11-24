#include <iostream>
#include <vector>
#include "include/parallelcore.cuh"

using namespace std;

struct Graph {
    int n_nodes;
    vector<int> R; // Row Offset
    vector<int> C; // Coalesced Adjacency Lists
};

vector<float> brandes(Graph G) {

    int *d_R, *d_C, *d_d, *d_sigma, *d_Q_curr, *d_Q_next, *d_S, *d_S_ends;
    float *d_delta, *d_CB;

    CUDA_ERR_CHK(cudaMalloc((void **) &d_R, G.R.size()*sizeof(int)));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_C, G.C.size()*sizeof(int)));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_d, G.n_nodes*sizeof(int)));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_sigma, G.n_nodes*sizeof(int)));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_Q_curr, G.n_nodes*sizeof(int)));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_Q_next, G.n_nodes*sizeof(int)));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_S, G.n_nodes*sizeof(int)));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_S_ends, (G.n_nodes + 1)*sizeof(int)));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_delta, G.n_nodes*sizeof(float)));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_CB, G.n_nodes*sizeof(float)));

    CUDA_ERR_CHK(cudaMemcpy(d_R, &(G.R[0]), G.R.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_ERR_CHK(cudaMemcpy(d_C, &(G.C[0]), G.C.size()*sizeof(int), cudaMemcpyHostToDevice));

    for(int i = 0; i < G.n_nodes; i++) {
        // cout << i << endl;
        brandes_parallel <<< 1, 1024>>> (i, d_R, d_C, d_d, d_sigma, d_delta, d_Q_curr, d_Q_next, d_S, d_S_ends, d_CB, G.n_nodes);
        CUDA_ERR_CHK(cudaPeekAtLastError());
        CUDA_ERR_CHK(cudaThreadSynchronize()); // Checks for execution error
    }

    vector<float> CB(G.n_nodes, 0);
    CUDA_ERR_CHK(cudaMemcpy(&(CB[0]), d_CB, G.n_nodes*sizeof(float), cudaMemcpyDeviceToHost));
    return CB;
}

int main() {
    int ex_R[] = {0, 3, 5, 8, 12, 16, 20, 24, 27, 28};
    int ex_C[] = {1, 2, 3, 0, 2, 0, 1, 3, 0, 2, 4, 5, 3, 5, 6, 7, 3, 4, 6, 7, 4, 5, 7, 8, 4, 5, 6, 6};
    
    // int ex_R[] = {0, 1, 3, 4};
    // int ex_C[] = {1, 0, 2, 1};
    
    // cout << sizeof(ex_C)/sizeof(ex_C[0]) << endl;

    Graph G;
    G.n_nodes = 9;
    G.R = vector<int>(ex_R, ex_R + sizeof(ex_R)/sizeof(ex_R[0]));
    G.C = vector<int>(ex_C, ex_C + sizeof(ex_C)/sizeof(ex_C[0]));

    vector<float> CB = brandes(G);

    int node = 0;
    for(auto i:CB) {
        cout << node << " : " << i << "\n";
        node ++;
    }
    
    cout << endl << endl;
    
    return 0;
}