#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <string>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

#include "genlib.hpp"

#define SYSERROR()  errno

using namespace std;

typedef unsigned char byte;

/*
    1 Step -> M Batches -> N nodes per batch
*/

void make_dir(string path) {
    const int dir_err = mkdir(path.c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (dir_err == -1)
    {
        cerr << "Error creating directory:" << path;
        exit(1);
    }
}

void check_make_dir(string path) {
    DIR* dir = opendir(path.c_str());

    if (dir) {
        closedir(dir);
    }
    else if(SYSERROR() = ENOENT) {
        make_dir(path);
    }
    else {
        cerr << "opendir failed";
        exit(1);
    }
}

void generate_file(string path, int nodes, opts vars) {
    
    ofstream fout_r;
    fout_r.open(path + "_r");

    ofstream fout_c;
    fout_c.open(path + "_c");

    int n_edges, i, j;
    if (fout_r.is_open() && fout_c.is_open()) {

        int curr = 0;
        fout_r << curr << endl;
        for (i = 0; i < nodes; i++) {
            n_edges = vars.minedges + rand()%(vars.maxedges - vars.minedges + 1);
            for(j = 0; j < n_edges; j++) {

                int e = rand()%nodes;
                while(e == i) e = rand()%nodes; 
                fout_c << (e) << endl;
            }
            curr += n_edges;
            fout_r << curr << endl;
        }

        fout_c.close();
        fout_r.close();
    }
    else {
        cerr << "Failed to open file : "<< SYSERROR() << std::endl;
        exit(1);
    }
}

void generate(opts vars) {
    int i, j, k, n_edges;
    string path;
    for(i = vars.n_nodes_start; i <= vars.n_nodes_end; i*=vars.step) {
        for (j = 0; j < vars.n_files; j++) {
            path = vars.path + "/" + to_string(i);
            check_make_dir(path);
            path = path + "/" + to_string(j);  
            cout << path << endl;

            generate_file(path, i, vars);
        }
    }
}

int main(int argc, char const *argv[])
{
    opts vars = get_defaults();
    
    srand(123456);
    check_make_dir(vars.path);
    generate(vars);

    return 0;
}
