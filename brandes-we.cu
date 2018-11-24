#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
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

    CUDA_ERR_CHK(cudaFree(d_R));
    CUDA_ERR_CHK(cudaFree(d_C));
    CUDA_ERR_CHK(cudaFree(d_d));
    CUDA_ERR_CHK(cudaFree(d_sigma));
    CUDA_ERR_CHK(cudaFree(d_Q_curr));
    CUDA_ERR_CHK(cudaFree(d_Q_next));
    CUDA_ERR_CHK(cudaFree(d_S));
    CUDA_ERR_CHK(cudaFree(d_S_ends));
    CUDA_ERR_CHK(cudaFree(d_delta));
    CUDA_ERR_CHK(cudaFree(d_CB));

    return CB;
}

int main() {
    Graph G;

    ofstream dump("dump.csv", ios::app);
    for(int nodevals = 2; nodevals <= 8; nodevals++) {

        ifstream f_c;
        ifstream f_r;
        f_c.open("ca_lab_graphs/c" + to_string(nodevals), ios::binary);
        f_r.open("ca_lab_graphs/r" + to_string(nodevals), ios::binary);

        int n;
        if(f_c && f_r){
            f_c.seekg(0, f_c.end);
            n = f_c.tellg();
            f_c.seekg(4, f_c.beg);

            G.C = vector<int> ( (n - 4)/4, 0);
            f_c.read( (char *) &(G.C[0]), n*sizeof(int));

            f_r.seekg(0, f_r.end);
            n = f_r.tellg();
            f_r.seekg(4, f_r.beg);

            G.R = vector<int> ( (n-4)/4, 0);
            f_r.read( (char*) &(G.R[0]), n*sizeof(int));

            f_c.close();
            f_r.close();
        }

        G.n_nodes = G.R.size() - 1;

        auto start = chrono::high_resolution_clock::now();
        vector<float> CB = brandes(G);
        auto end = chrono::high_resolution_clock::now();

        auto _time = chrono::duration_cast<chrono::milliseconds>(end - start);

        ofstream fout("ca_lab_graphs/pll_a" + to_string(nodevals));

        float maxBC = -1;
        int maxNode = -1;

        int node = 0;
        for(auto i:CB) {
            if(i > maxBC) {
                maxBC = i;
                maxNode = node;
            }
            fout << node << " : " << i << "\n";
            node ++;
        }
    
        fout << endl;
        fout.close();

        cout << "PLL," << G.n_nodes << "," << _time.count() << "ms" << endl;
        dump << "PLL," << G.n_nodes << "," << _time.count() << endl;

        cout << "BC Node = " << maxNode << ", BC = " << maxBC << endl;

    }
    dump.close();
    return 0;
}