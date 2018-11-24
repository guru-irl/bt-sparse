#ifndef PARALLELCORE_CUH
#define PARALLELCORE_CUH

#include <cuda.h>
#include <iostream>
#include <math.h>

#define CUDA_ERR_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

using namespace std;

__device__ void find_shortest_paths( 
    int *R, int *C, int *d, int *sigma, int *Q_curr, int *Q_next, int *S, int *S_ends, 
    int &Q_curr_len, int &Q_next_len, int &S_len, int &S_ends_len, int &depth) {

    int id = threadIdx.x;

    // printf("%i \n", id);

    int bsize = blockDim.x;
    int i, j, v, w, last;
    bool done = false;
    while(!done) {
        for(i = id; i < Q_curr_len; i += bsize) {
            v = Q_curr[i];
            for(j = R[v]; j < R[v+1]; j++) {
                w = C[j];
                if(atomicCAS(&d[w], -1, d[v] + 1) < 0) {
                    last = atomicAdd(&Q_next_len, 1);
                    Q_next[last] = w;
                }
                if(d[w] == (d[v] + 1)) {
                    atomicAdd(&sigma[w], sigma[v]);
                }
            }
        }

        __syncthreads();

        if(Q_next_len == 0) {
            if(id == 0) {
                depth = d[S[S_len-1]];
                /*
                printf ("    tid %d zero Queue next %d : ", id, Q_next_len);
                for(i = 0; i < Q_next_len; i++) {
                    printf("%d ", Q_next[i]);
                }
                // printf ("\n");
                printf ("\n    New Depth : %d \n", d[S[S_len-1]]);*/
            }

            __syncthreads();
            done = true;
        }
        if ( id >= 0 && Q_next_len != 0) {

            // __syncthreads();

            for(i = id; i < Q_next_len; i += bsize) {
                Q_curr[i] = Q_next[i];
                // printf("    i: %d %d %d \n", i, Q_next[i], Q_curr[i]);
                S[i + S_len] = Q_next[i];
            }

            __syncthreads();

            if(id == 0) {
                S_ends[S_ends_len] = S_ends[S_ends_len-1] + Q_next_len;
                S_ends_len = S_ends_len + 1;
                Q_curr_len = Q_next_len;
                S_len = S_len + Q_next_len;
                
                /*printf ("    tid %d else Queue next %d : ", id, Q_next_len);
                for(i = 0; i < Q_next_len; i++) {
                    printf("%d ", Q_next[i]);
                }
                printf ("\n");

                printf("    Stack : ");
                for(i = 0; i < S_len; i++) {
                    printf("%d ", S[i]);
                }
                printf("\n");

                printf("    Stack ends : ");
                for(i = 0; i < S_ends_len; i++) {
                    printf("%d ", S_ends[i]);
                }
                printf("\n");
                */
                Q_next_len = 0;
            }

            __syncthreads();
        }
    }
}

__device__ void accumulate_dependencies( int *R, int *C, int *d, int *sigma, float *delta, int *S, int *S_ends, int &depth) {

    int id = threadIdx.x;
    int bsize = blockDim.x;
    int i, j, v, w;
    float sw, sv, dsw;

    while(depth > 0) {
        for (i = id + S_ends[depth]; i < S_ends[depth+1]; i += bsize) {
            w = S[i];
            dsw = 0;
            sw = sigma[w];

            for(j = R[w]; j < R[w+1]; j++) {
                v = C[j];
                sv = sigma[v];
                if(d[v] == d[w] + 1) {
                    dsw += ((sw/sv)*(1+delta[v]));
                }
            }

            delta[w] = dsw;
        }

        __syncthreads();
        if(id == 0) depth--;
        __syncthreads();
    }
}

__global__ void brandes_parallel(int d_s, int *R, int *C, int *d, int *sigma, float *delta, int *Q_curr, int *Q_next, int *S, int *S_ends, float *CB, int n_nodes) {

    int id = threadIdx.x;
    // int s = blockIdx.x;
    __shared__ int s;
    __shared__ int bsize;
    __shared__ int Q_curr_len;
    __shared__ int Q_next_len;
    __shared__ int S_len;
    __shared__ int S_ends_len;
    __shared__ int depth;

    if(id == 0) {
        s = d_s;
        bsize = blockDim.x;

        /*for(int i = 0; i < n_nodes; i++) {
            printf("    Node %d : ", i);
            for(int j = R[i]; j < R[i+1]; j++) {
                printf("%d ", C[j]);
            }
            printf("\n");
        }*/

    }

    __syncthreads();

    int v;
    for(v = id; v < n_nodes; v += bsize) {
        if(s == 0) {
            CB[v] = 0;
        }
        if(v == s) {
            d[v] = 0;
            sigma[v] = 1;
        }
        else {
            d[v] = -1;
            sigma[v] = 0;
        }
        delta[v] = 0;
    }

    if(id == 0) {
        Q_curr[0] = s;
        Q_curr_len = 1;
        Q_next_len = 0;
        S[0] = s;
        S_len = 1;
        S_ends[0] = 0;
        S_ends[1] = 1;
        S_ends_len = 2;
        depth = 0;

        // printf("Init done\n");
    }

    __syncthreads();
    
    /*if(id == 0) {
        printf("    |  D  | Sigma | Delta |\n");
        for(v = 0; v < n_nodes; v++) {
            printf("    |%5d|%7d|%7d|\n", d[v], sigma[v], delta[v]);
        }
        printf ("\n\n");
    }

    __syncthreads();*/

    find_shortest_paths( R, C, d, sigma, Q_curr, Q_next, S, S_ends, Q_curr_len, Q_next_len, S_len, S_ends_len, depth/*, id, bsize*/);
    __syncthreads();

    /*if(id == 0) {
        int d, e;
        for(d = 0; d <= depth; d++) {
            printf("    Level %d : ", d);
            for(e = S_ends[d]; e < S_ends[d+1]; e++) {
                printf("%d ", S[e]);
            }
            printf("\n");
        }
    }*/

    // if(id == 0) printf("fsp done\n");

    /*if(id == 0) {
        printf("    |  i  |  D  | Sigma | Delta |\n");
        for(v = 0; v < n_nodes; v++) {
            printf("    |%5d|%5d|%7d|%7d|\n", v, d[v], sigma[v], delta[v]);
        }
        printf ("\n\n");
    }*/

    accumulate_dependencies(R, C, d, sigma, delta, S, S_ends, depth/*, id, bsize*/);
    __syncthreads();
    // if(id == 0) printf("ad done\n");

    /*if(id == 0) {
        printf("    |  i  |  D  | Sigma | Delta |\n");
        for(v = 0; v < n_nodes; v++) {
            printf("    |%5d|%5d|%7d|%7d|\n", v, d[v], sigma[v], delta[v]);
        }
        printf ("\n\n");
    }*/

    for(v = id; v < n_nodes; v += bsize) {
        if(id != s) {
            CB[v] += delta[v]/2;
        }
    }
}

#endif