#include <bits/stdc++.h>
#include <iostream>

using namespace std;

struct Graph {
    int n_nodes;
    vector<int> R; // Row Offset
    vector<int> C; // Coalesced Adjacency Lists
};

/*
* C[R[0]] to C[R[1]] holds the adjacency list for 
* node 0. Not inclusive.
*/

vector<float> brandes(Graph G) {

    vector<float> CB(G.n_nodes, 0);
    for(int s = 0; s < G.n_nodes; s++) {

        stack<int> S;
        vector<vector<int> > P(G.n_nodes);
        vector<int> sigma(G.n_nodes, 0); sigma[s] = 1;
        vector<int> d(G.n_nodes, -1); d[s] = 0;

        queue<int> Q;
        Q.push(s);

        int v, w, i;
        while(!Q.empty()) {
            v = Q.front(); Q.pop();
            S.push(v);

            for (i = G.R[v]; i < G.R[v+1]; i++) {
                w = G.C[i];

                if(d[w] < 0) {
                    Q.push(w);
                    d[w] = d[v] + 1;
                }

                if(d[w] == d[v] + 1) {
                    sigma[w] += sigma[v];
                    P[w].push_back(v);
                }
            }
        }

        vector<float> delta(G.n_nodes, 0);

        while(!S.empty()) {
            w = S.top(); S.pop();
            for(i = 0; i < P[w].size(); i++) {
                v = P[w][i];
                delta[v] += (((float)sigma[v]/(float)sigma[w])*(1+delta[w]));
            }
            if(w != s) CB[w] += (delta[w]/2.0);
        }
    }

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