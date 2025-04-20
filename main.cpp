#include <iostream>
#include "src/MLP.h"

int main() {
    // MLP CLASSIFICATION

    std::cout << "MLP CLASSIFICATION" << std::endl;

    MLP model = MLP({2, 1});

    std::vector<std::vector<double> > dataset_inputs = {
        {0., 0.},
        {1., 1.},
        {0., 1.}
    };

    std::vector<std::vector<double> > dataset_expected_outputs = {
        {1.},
        {-1.},
        {-1.}
    };


    model.train(dataset_inputs, dataset_expected_outputs, 1000000, 0.1, true);

    for (int i = 0; i < dataset_inputs.size(); i++) {
        std::vector<double> result = model.predict(dataset_inputs[i], true);
        for (int j = 0; j < result.size(); j++) {
            std::cout << result[j] << std::endl;
        }
    }


    // MLP REGRESSION

    std::cout << "MLP REGRESSION" << std::endl;

    model = MLP({2, 1});

    dataset_inputs = {
        {0.},
        {1.},
        {2.}
    };

    dataset_expected_outputs = {
        {40},
        {90},
        {120}
    };


    model.train(dataset_inputs, dataset_expected_outputs, 1000000, 0.1, false);

    for (int i = 0; i < dataset_inputs.size(); i++) {
        std::vector<double> result = model.predict(dataset_inputs[i], false);
        for (int j = 0; j < result.size(); j++) {
            std::cout << result[j] << std::endl;
        }
    }
}
