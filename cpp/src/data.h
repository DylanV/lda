/*!
 \file data.h
 */

#ifndef LDA_DATA_H
#define LDA_DATA_H

#include <string>
#include <map>
#include <vector>
#include <fstream>

#include "dirichlet.h"
#include "lda_model.h"

//! load a corpus from a file
doc_corpus load_corpus(std::string file_path);
//! split a string with the given delimiter
std::vector<std::string> split(std::string const& str, char delim);
//! load vocab from file
std::vector<std::string> load_vocab(std::string file_path);
//! load settings from file
void load_settings(std::string file_path, alpha_settings& alpha, lda_settings& lda);


//! write vector to file stream
template <typename T>
void write_vector_to_fs(std::fstream& fs, const std::vector<T>& vec) {
    char sep = ' ';
    for(const T& val : vec){
        fs << val;
        if(&val != &vec.back()){
            fs << sep;
        }
    }
}

//! write 2d vector to file stream
template <typename T>
void write_2d_vector_to_fs(std::fstream& fs, const std::vector<std::vector<T> >& vec) {
    char nl = '\n';
    char sep = ' ';

    if(fs.is_open()){
        for(const std::vector<T>& inner_vec : vec ){
            write_vector_to_fs(fs, inner_vec);
            if(&inner_vec != &vec.back()){
                fs << nl;
            }
        }
    }
}

#endif //LDA_DATA_H
