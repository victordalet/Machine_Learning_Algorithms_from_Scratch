#include <iostream>

#include "src/Linear.h"
#include "src/MLP.h"

int main() {
    // MLP CLASSIFICATION

    std::cout << "MLP CLASSIFICATION" << std::endl;

    auto model = MLP({2, 1});

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

    for (const auto &dataset_input: dataset_inputs) {
        for (std::vector<double> result = model.predict(dataset_input, true); const double j: result) {
            std::cout << j << std::endl;
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

    for (const auto &dataset_input: dataset_inputs) {
        std::vector<double> result;
        result = model.predict(dataset_input, false);
        for (const auto j: result) {
            std::cout << j << std::endl;
        }
    }

    // Linear REGRESSION

    std::cout << "Linear REGRESSION" << std::endl;

    auto model_linear = Linear();

    std::vector<double> x_data = {4, 5, 6};

    std::vector<double> y_data = {4, 5, 6};

    model_linear.train(x_data, y_data, 100, 0.01, false);

    std::cout << model_linear.predict(50, false) << std::endl;


    // Linear CLASSIFICATION

    std::cout << "Linear CLASSIFICATION" << std::endl;

    model_linear = Linear();

    x_data = {4, 2, 6};

    y_data = {0, 0, 1};

    model_linear.train(x_data, y_data, 100, 0.01, true);

    std::cout << model_linear.predict(50, true) << std::endl;
}
