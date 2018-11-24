#include <bits/stdc++.h>
#include <chrono>
#include <iostream>

#define isUnidrected false

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
            if(w != s){
                if(!isUnidrected) 
                    CB[w] += (delta[w]);
                else 
                    CB[w] += (delta[w]/2);                
            }
        }
    }

    return CB;
}

int main() {

    Graph G;
    ofstream dump("dump.csv", ios::app);

    for(int nodevals = 2; nodevals <= 8; nodevals++) {
        
        ifstream f_c;
        ifstream f_r;

        // cout << "ca_lab_graphs/c" + to_string(nodevals);

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
        // cout << G.n_nodes << endl; 
        // cout << G.C.size() << endl;

        auto start = chrono::high_resolution_clock::now();
        vector<float> CB = brandes(G);
        auto end = chrono::high_resolution_clock::now();

        auto _time = chrono::duration_cast<chrono::milliseconds>(end - start);

        ofstream fout("ca_lab_graphs/seq_a" + to_string(nodevals));

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
    

        fout << endl << endl;
        fout.close();

        cout << "SEQ," << G.n_nodes << "," << _time.count() << "ms" << endl;
        dump << "SEQ," << G.n_nodes << "," << _time.count() << endl;

        cout << "BC Node = " << maxNode << ", BC = " << maxBC << endl;

    }
    dump.close();
    return 0;
}

/*
    int ex_R[] = {0, 3, 5, 8, 12, 16, 20, 24, 27, 28};
    int ex_C[] = {1, 2, 3, 0, 2, 0, 1, 3, 0, 2, 4, 5, 3, 5, 6, 7, 3, 4, 6, 7, 4, 5, 7, 8, 4, 5, 6, 6};
    
    int ex_R[] = {0, 1, 3, 4};
    int ex_C[] = {1, 0, 2, 1};
    
    cout << sizeof(ex_C)/sizeof(ex_C[0]) << endl;
    Graph G;
    G.n_nodes = 9;
    G.R = vector<int>(ex_R, ex_R + sizeof(ex_R)/sizeof(ex_R[0]));
    G.C = vector<int>(ex_C, ex_C + sizeof(ex_C)/sizeof(ex_C[0]));
*/  