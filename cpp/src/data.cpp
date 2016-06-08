//
// Created by Dylan on 14/04/2016.
//

#include <sstream>
#include <fstream>
#include <iostream>
#include <set>
#include <math.h>
#include <string>
#include <algorithm>
#include "data.h"

std::vector<std::string> split(std::string const& str, char delim)
{
    std::vector<std::string> items;
    std::stringstream ss(str);
    std::string item;

    while (std::getline(ss, item, delim)) {
        if(!item.empty()){ items.push_back(item); }
    }

    return items;
}

doc_corpus load_corpus(std::string file_path)
{
    std::ifstream fs(file_path);

    const char line_delim = ' ';
    const char item_delim = ':';
    doc_corpus corpus;
    std::vector<document> docs;

    if(fs.is_open()){
        int doc_count = 0;
        std::string line;

        std::set<int> vocab;

        while(!fs.eof()){
            getline(fs, line);
            if(line != ""){
                std::vector<std::string> items = split(line, line_delim);

                int unique_count = stoi(items[0]);
                items.erase(items.begin());

                document doc;
                std::map<int, int> word_counts;
                int count =0;

                for(std::string const& str : items){
                    if(str != "\r" && str != "\n"){
                        std::vector<std::string> wc_str = split(str, item_delim);
                        int word_id = stoi(wc_str[0]);
                        int word_count = stoi(wc_str[1]);

                        vocab.insert(word_id);
                        word_counts[word_id] += word_count;
                        count += word_count;
                    }
                }

                if(word_counts.size() != unique_count){
                    std::cout << "The corpus lies" << std::endl;
                }

                doc.wordCounts = word_counts;
                doc.count = count;
                doc.uniqueCount = unique_count;
                doc_count++;
                docs.push_back(doc);
            }
        }
        corpus.numTerms = vocab.size();
        corpus.numDocs = doc_count;
    }

    corpus.docs = docs;
    return corpus;
}

std::vector<std::string> load_vocab(std::string file_path)
{
    std::ifstream fs(file_path);
    std::vector<std::string> vocab;

    if(fs.is_open()){
        std::string line;
        while(!fs.eof()){
            getline(fs, line);
            vocab.push_back(line);
        }
    }
    return vocab;
}

void load_settings(std::string file_path, alpha_settings& alpha, lda_settings& lda){

    std::ifstream fs(file_path);
    const char line_delim = ' ';
    int numAlpha = 0;
    int numLDA = 0;
    bool loadingLDA = false;
    bool loadingAlpha = false;

    if(fs.is_open()){
        std::string line;
        while(!fs.eof()){
            getline(fs, line);
            if(line != ""){
                std::vector<std::string> items = split(line, line_delim);
                if(items.size() == 2){

                    std::string value = items[1];
                    value.erase(std::remove(value.begin(), value.end(), '\n'), value.end());
                    value.erase(std::remove(value.begin(), value.end(), '\r'), value.end());
                    value.erase(std::remove(value.begin(), value.end(), ' '), value.end());

                    if(items[0] == "LDA"){
                        numLDA = std::stoi(items[1]);
                        loadingLDA = true;
                        loadingAlpha = false;
                    }

                    else if(items[0] == "ALPHA"){
                        numAlpha = std::stoi(items[1]);
                        loadingLDA = false;
                        loadingAlpha = true;
                    }

                    if(loadingLDA){
                        if(items[0] == "converged_threshold")
                            lda.converged_threshold = std::stod(value);
                        else if(items[0] == "min_iterations")
                            lda.min_iterations = std::stoi(value);
                        else if(items[0] == "max_iterations")
                            lda.max_iterations = std::stoi(value);
                        else if(items[0] == "inf_converged_threshold")
                            lda.inf_converged_threshold = std::stod(value);
                        else if(items[0] == "inf_max_iterations")
                            lda.inf_max_iterations = std::stoi(value);
                    }

                    if(loadingAlpha){
                        if(items[0] == "estimate_alpha")
                            alpha.estimate_alpha = (value == "true");
                        else if(items[0] == "concentration")
                            alpha.concentration = (value == "true");
                        else if(items[0] == "newton_threshold")
                            alpha.newton_threshold = std::stod(value);
                        else if(items[0] == "max_iterations")
                            alpha.max_iterations = std::stoi(value);
                        else if(items[0] == "init_prec")
                            alpha.init_prec = std::stod(value);
                        else if(items[0] == "init_s")
                            alpha.init_s = std::stoi(value);
                    }
                }
            }
        }
    }
}

void write_parameters_to_file(std::string param_dir, const lda& model) {
    /*!
    Writes the lda alpha, beta and gamma dirchlet parameters to files.
    File contains the vector as a space seperated array. Each row on a new line.
    /param folder_path the path to the folder where the parameter files will be written.
 */
    std::fstream fs; //get the filestream

    //get beta from logProbW
    std::vector<std::vector<double> > beta
            = std::vector<std::vector<double> >(model.numTopics, std::vector<double>(model.numTerms, 0));
    for(int k=0; k<model.numTopics; k++){
        for(int n=0; n<model.corpus.numTerms; n++){
            beta[k][n] = exp(model.logProbW[k][n]);
        }
    }
    //write beta
    fs.open(param_dir+"beta.dat", std::fstream::out | std::fstream::trunc);
    write_2d_vector_to_fs(fs, beta);
    fs.close();

    //write gamma
    fs.open(param_dir+"gamma.dat", std::fstream::out | std::fstream::trunc);
    write_2d_vector_to_fs(fs, model.varGamma);
    fs.close();

    //write alpha
    fs.open(param_dir+"alpha.dat", std::fstream::out | std::fstream::trunc);
    write_vector_to_fs(fs, model.alpha.alpha);
    fs.close();
}

//void lda::loadFromParams(std::string folder_path)
//{
///*!
//    /param folder_path the path to the folder containing the parameter files
//    /sa writeParams()
// */
//    char sep = ' ';
//    logProbW = loadBetaFromFile(folder_path + "beta.dat");
//}
//
//std::vector<std::vector<double>> lda::loadBetaFromFile(std::string file_path) {
//
//    std::vector<std::vector<double>> beta;
//    char sep = ' ';
//
//    std::ifstream fs(file_path);
//    if(fs.is_open()){
//        std::string line;
//        bool readFirst = false;
//
//        while(!fs.eof()){
//            getline(fs, line);
//            std::vector<std::string> items = split(line, sep);
//
//            if(!readFirst){
//                assert(items.size() == 2);
//                readFirst = true;
//                numTopics = stoi(items[0]);
//                numTerms = stoi(items[1]);
//                assert(numTerms == corpus.numTerms);
//            } else{
//                if(line != ""){
//                    assert(items.size() == numTerms);
//                    std::vector<double> top_probs;
//                    for(auto const& item : items){
//                        top_probs.push_back(stod(item));
//                    }
//                    beta.push_back(top_probs);
//                }
//            }
//        }
//    }
//
//    return beta;
//}