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

/*! Load a corpus from file
 * Load a bag of words style corpus from a file. Each document on a new line. Document format is as follows:
 * <numver of unique terms> <word id>:<count> <word id>:<count>
 * Word ids are assumed continuos and zero indexed.
 * @param file_path Path to file
 * @return Document corpus struct. @sa doc_corpus
 */
doc_corpus load_corpus(std::string file_path);

/*! Split a string with a given delimiter
 * @param str String to be split
 * @param delim Delimiter
 * @return Vector of substrings
 */
std::vector<std::string> split(std::string const& str, char delim);

/*! Load a vocabulary file
 * @param [in] file_path
 * @return Vector of words in vocabulary
 */
std::vector<std::string> load_vocab(std::string file_path);

/*! Loads settings from a file into a map
 * Takes a standard settings file and loads each line into a string pair in a map.
 * Map can be used to set up settings object for the relevent inference method.
 * @param file_path [in] The path to the settings file.
 * @return String to string map of settings names and values.
 */
std::map<std::string, std::string> load_settings_file(const std::string file_path);

/*! Write a vector to the given file stream.
 * Writes the vector to a single line in the given file stream. Items are space seperated.
 * @param [in] fs File stream.
 * @param [in] vec Vector to be written.
 */
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

/*! Write a 2d vector to a filestream.
 * Each vector is written to a line. Items are space seperated.
 * @param [in,out] fs File stream.
 * @param [in] vec Vector of vectors to be written.
 */
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
