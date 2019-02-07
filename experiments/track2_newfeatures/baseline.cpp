#include <iostream>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <vector>
#include <limits>

#include "./parser.h"
#include "ripped_evaluator/evaluator.h"

int main() {
    // Fast read
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    // Model init
    const std::string MODEL_FILE = "track_2_model.cbm";
    NCatboostStandalone::TOwningEvaluator evaluator(MODEL_FILE);

    // Skip header
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    std::cout << "id,prediction\n";

    // Read and predict
    while (std::cin.good() && std::cin.peek() != EOF) {
	    std::vector<float> features(N_FEATURES);
        size_t id;
        ugly_hardcoded_parse(std::cin, &id, &features);
        const float prediction = \
            evaluator.Apply(features, NCatboostStandalone::EPredictionType::RawValue);
        std::cout << id << DELIMITER << prediction  << '\n';
    }
    return 0;

    /*
        0 - Features
        1 - Catboost
        2 - LightGBM
        3 - Nearest Centroid
        4 - Random Forest
        5 - RidgeClassifier
    */
}
