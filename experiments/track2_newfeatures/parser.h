// Copyright 2019, Nikita Kazeev, Higher School of Economics
#pragma once
#include <ctype.h>
#include <cstdint>
#include <cmath>

#include <iostream>
#include <vector>
#include <array>
#include <limits>

const size_t N_STATIONS               = 4;
const size_t FOI_FEATURES_PER_STATION = 6;

// The structure of .csv is the following:
// id, <62 float features>, number of hits in FOI, <9 arrays of FOI hits features>, <2 float features>

/* Number of features */
const size_t N_RAW_FEATURES           = 65;
const size_t N_RAW_FEATURES_TAIL      = 2;
const size_t N_ADDITIONAL_FEATURES    = 28;

const size_t N_FEATURES               = N_RAW_FEATURES + N_STATIONS * FOI_FEATURES_PER_STATION + N_ADDITIONAL_FEATURES;
const size_t ADD_FEATURES_START       = N_RAW_FEATURES + N_STATIONS * FOI_FEATURES_PER_STATION;
const size_t FOI_FEATURES_START       = N_RAW_FEATURES;

/* Indices for different type of features*/
const size_t FOI_HITS_N_INDEX         = 62;

const size_t LEXTRA_X_INDEX           = 45;
const size_t LEXTRA_Y_INDEX           = 49;

const size_t MATCHED_HIT_X_IND        = 15;
const size_t MATCHED_HIT_Y_IND        = 19;
const size_t MATCHED_HIT_Z_IND        = 23;

const float EMPTY_FILLER              = 1000;

const char DELIMITER = ',';

void ugly_hardcoded_parse(std::istream& stream, size_t* id, std::vector<float>* result);
