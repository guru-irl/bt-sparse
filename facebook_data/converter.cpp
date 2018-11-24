#include <iostream>
#include <fstream> 
#include <string>
#include <vector>
#include <string.h>

using namespace std; 

int main()
{
    int count = 0;
    vector<vector<int> > data;
    string line;

    ifstream myfile ("Facebook_Data.csv");
    if (myfile.is_open()){

        vector<int> row;
        while(!myfile.eof()){

            getline(myfile, line);
            for(auto i = line.begin() ; i != line.end() ; ++i){

                char *a = &(*i);
                if(*i == '1' || *i =='0')
                    row.push_back(atoi(a));
            }
            data.push_back(row);
        }
        myfile.close();
     }
     else
        cout << "Unable to open file." << endl;

    // now we have the matrix
    // convert it into a CSR

    vector<int> R;
    vector<int> C;
    int n;

    R.push_back(0);
    for(int i = 0 ; i < data.size() ; ++i){
        for(int j = i ; j < data[i].size() ; ++j){

            if(data[i][j]==1){

                C.push_back(j);
                ++n;
            }
        }
        R.push_back(n);
    }
    R.push_back(n+1);

    

    return 0;
}