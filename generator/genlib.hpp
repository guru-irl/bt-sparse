#ifndef GENLIB_H
#define GENLIB_H

#include <string>

struct opts {
    int n_nodes_start;
    int n_nodes_end;
    int n_files;
    int step;
    std::string path;
    int minedges;
    int maxedges;
};

opts get_defaults(){
    opts default_vars;
    default_vars.path = "../dataset";
    default_vars.n_nodes_start = 100;
    default_vars.n_nodes_end = 100000;
    default_vars.n_files = 3;
    default_vars.step = 10; // Multiplied
    default_vars.minedges = 1; 
    default_vars.maxedges = 5; 

    return default_vars;
}
#endif