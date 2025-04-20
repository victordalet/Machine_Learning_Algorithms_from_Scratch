//
// Created by victor on 21/04/25.
//

#include "Linear.h"

#include <iostream>
#include <ostream>
#include <random>
#include <math.h>


Linear::Linear() {
    this->m = (rand() % 2000 - 1000) / 1000.0;
    this->b = (rand() % 2000 - 1000) / 1000.0;
}


void Linear::train(const std::vector<double> &x_data, const std::vector<double> &y_data, const int epochs,
                   const double learning_rate, const bool is_classification) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double m_gradient = 0.;
        double b_gradient = 0.;

        const double n = x_data.size();

        for (int i = 0; i < x_data.size(); i++) {
            double y_pred = this->m * x_data[i] + this->b;
            if (is_classification) y_pred = sigmoid(y_pred);
            m_gradient += -2 * x_data[i] * (y_data[i] - y_pred);
            b_gradient += -2 * (y_data[i] - y_pred);
        }

        this->m -= (m_gradient / n) * learning_rate;
        this->b -= (b_gradient / n) * learning_rate;
    }
    std::cout << this->m << std::endl;
    std::cout << this->b << std::endl;
}


double Linear::sigmoid(const double x) {
    return 1 / std::exp(-x) + 1;
}

double Linear::predict(const double x_data, const bool is_classification) const {
    return is_classification ? sigmoid(this->m * x_data + this->b) : this->m * x_data + this->b;
}
