/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 *
 * Code from napkinXML
 * https://github.com/mwydmuch/napkinXML
 */

#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <thread>
#include <iostream>
#include <algorithm>

#include "real.h"
#include "smatrix.h"

namespace fasttext {

struct Feature {
    int index;
    real value;


    // In extremeText we do not need this kind of sorting
    /*
    bool operator<(const Feature& r) const { return value < r.value; }
    bool operator>(const Feature& r) const { return value > r.value; }
    */

    bool operator<(const Feature& r) const { return index < r.index; }
    bool operator>(const Feature& r) const { return index > r.index; }

    friend std::ostream& operator<<(std::ostream& os, const Feature& fn) {
        os << fn.index << ":" << fn.value;
        return os;
    }
};

// Sparse utils

template <typename T, typename U>
inline T argMax(const std::unordered_map<T, U>& map){
    auto pMax = std::max_element(map.begin(), map.end(),
                                 [](const std::pair<T, U>& p1, const std::pair<T, U>& p2)
                                 { return p1.second < p2.second; });
    return pMax.first;
}

template <typename T, typename U>
inline T argMin(const std::unordered_map<T, U>& map){
    auto pMin = std::min_element(map.begin(), map.end(),
                                 [](const std::pair<T, U>& p1, const std::pair<T, U>& p2)
                                 { return p1.second < p2.second; });
    return pMin.first;
}


template <typename T>
inline size_t argMax(const std::vector<T>& vector){
    return std::distance(vector.begin(), std::max_element(vector.begin(), vector.end()));
}

template <typename T>
inline size_t argMin(const std::vector<T>& vector){
    return std::distance(vector.begin(), std::min_element(vector.begin(), vector.end()));
}

// Sparse vector dot dense vector
template <typename T>
inline T dotVectors(Feature* vector1, const T* vector2, const int& size){
    T val = 0;
    Feature* f = vector1;
    while(f->index != -1 && f->index < size) {
        val += f->value * vector2[f->index];
        ++f;
    }
    return val;
}

template <typename T>
inline T dotVectors(Feature* vector1, const T* vector2){ // Version without size checks
    T val = 0;
    Feature* f = vector1;
    while(f->index != -1) {
        val += f->value * vector2[f->index];
        ++f;
    }
    return val;
}

template <typename T>
inline T dotVectors(Feature* vector1, const std::vector<T>& vector2){
    //dotVectors(vector1, vector2.data(), vector2.size());
    dotVectors(vector1, vector2.data());
}

// Sets values of a dense vector to values of a sparse vector
template <typename T>
inline void setVector(Feature* vector1, T* vector2, size_t size, int shift = 0){
    Feature* f = vector1;
    while(f->index != -1 && f->index + shift < size){
        vector2[f->index + shift] = f->value;
        ++f;
    }
}

template <typename T>
inline void setVector(Feature* vector1, T* vector2, int shift = 0){ // Version without size checks
    Feature* f = vector1;
    while(f->index != -1){
        vector2[f->index + shift] = f->value;
        ++f;
    }
}

template <typename T>
inline void setVector(Feature* vector1, std::vector<T>& vector2, int shift = 0) {
    //setVector(vector1, vector2.data(), vector2.size(), shift);
    setVector(vector1, vector2.data(), shift);
}

// Zeros selected values of a dense vactor
template <typename T>
inline void setVectorToZeros(Feature* vector1, T* vector2, size_t size, int shift = 0){
    Feature* f = vector1;
    while(f->index != -1 && f->index + shift < size){
        vector2[f->index + shift] = 0;
        ++f;
    }
}

template <typename T>
inline void setVectorToZeros(Feature* vector1, T* vector2, int shift = 0){ // Version without size checks
    Feature* f = vector1;
    while(f->index != -1){
        vector2[f->index + shift] = 0;
        ++f;
    }
}

template <typename T>
inline void setVectorToZeros(Feature* vector1, std::vector<T>& vector2, int shift = 0) {
    //setVectorToZeros(vector1, vector2.data(), vector2.size());
    setVectorToZeros(vector1, vector2.data(), shift);
}

// Adds values of sparse vector to dense vector
template <typename T>
inline void addVector(Feature* vector1, T* vector2, size_t size){
    Feature* f = vector1;
    while(f->index != -1 && f->index < size){
        vector2[f->index] += f->value;
        ++f;
    }
}

template <typename T>
inline void addVector(Feature* vector1, std::vector<T>& vector2) {
    addVector(vector1, vector2.data(), vector2.size());
}

// Unit norm
template <typename T>
inline void unitNorm(T* data, size_t size){
    T norm = 0;
    for(int f = 0; f < size; ++f) norm += data[f] * data[f];
    norm = std::sqrt(norm);
    if(norm == 0) return;
    else for(int f = 0; f < size; ++f) data[f] /= norm;
}

inline void unitNorm(Feature* data, size_t size){
    real norm = 0;
    for(int f = 0; f < size; ++f) norm += data[f].value * data[f].value;
    norm = std::sqrt(norm);
    if(norm == 0) return;
    else for(int f = 0; f < size; ++f) data[f].value /= norm;
}

template <typename T>
inline void unitNorm(std::vector<T>& vector){
    unitNorm(vector.data(), vector.size());
}

}
