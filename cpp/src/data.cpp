//
// Created by Dylan on 14/04/2016.
//

#include <sstream>
#include <fstream>
#include <iostream>
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
