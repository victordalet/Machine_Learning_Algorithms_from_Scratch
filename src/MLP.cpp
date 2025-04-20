//
// Created by victor on 20/04/25.
//

#include "MLP.h"
#include <cmath>
#include <random>


MLP::MLP(const std::vector<int> &npl) {

    this->npl = npl;
    this->weights = {};
    this->L = npl.size() - 1;
    this->X = {};
    this->deltas = {};

    for (int l = 0; l < this->L + 1; l++) {
        this->weights.emplace_back();

        if (l == 0) continue;

        for (int i = 0; i < this->npl[l - 1] + 1; i++) {
            this->weights[l].emplace_back();
            for (int j = 0; j < this->npl[l] + 1; j++) {
                if (j == 0)
                    this->weights[l][i].push_back(0.0);
                else
                    this->weights[l][i].push_back((rand() % 2000 - 1000) / 1000.0);
            }
        }
    }

    for (int l = 0 ; l < this->L + 1 ; l++) {
        this->X.emplace_back();
        this->deltas.emplace_back();
        for (int j = 0 ; j < this->npl[l] + 1; j++) {
            this->deltas[l].push_back(0.);
            if (j==0)
                this->X[l].push_back(1.);
            else
                this->X[l].push_back(0.);

        }
    }


}


void MLP::propagate(const std::vector<double> &inputs, const bool is_classification) {
    for ( int i = 1 ; i < this->npl[0] + 1 ; i++) {
        this->X[0][i] = inputs[i -1];
    }

    for (int l = 1 ; l < this->L + 1 ; l++) {
        for (int j = 1 ; j < this->npl[l] + 1; j++) {
            double total = 0.;
            for (int i = 0 ; i < this->npl[l - 1] + 1; i++) {
                total += this->weights[l][i][j] * this->X[l-1][i];
            }
            if (is_classification) total = std::tanh(total);
            this->X[l][j] = total;
        }
    }
}

std::vector<double> MLP::predict(const std::vector<double> &input, const bool is_classification) {
      this->propagate(input,is_classification);
      return this->X[this->L];
}

void MLP::train(const std::vector<std::vector<double>> &all_dataset_inputs,
        const std::vector<std::vector<double>> &all_dataset_outputs,
        const int interations_count,
        const double learning_rate,
        const bool is_classification
    ) {
    for (int it = 0 ; it < interations_count ; it++) {
        const double k = rand() % all_dataset_inputs.size() -1;
        std::vector<double> sample_inputs = all_dataset_inputs[k];
        std::vector<double> sample_expected_outputs = all_dataset_outputs[k];

        this->propagate(sample_inputs, is_classification);

        for (int j = 1 ; j < this->npl[this->L] + 1 ; j++) {
            this->deltas[this->L][j] = this->X[this->L][j] - sample_expected_outputs[j - 1];
            if (is_classification)
                this->deltas[this->L][j] *= (1 - std::pow(this->X[this->L][j],2));

        }

        for (int l = this->L + 1 ; l > 2 ; l--) {
            for (int i = 1 ; i < this->npl[l-1] + 1; i++) {
                double total = 0.;
                for (int j = 1 ; j < this->npl[l] + 1 ; j++) {
                    total += this->weights[l][i][j] * this->deltas[l][j];
                }
                total *= (1 - std::pow(this->X[l-1][i],2));
                this->deltas[l-1][i] = total;
            }
        }

        for (int l = 1 ; l < this->L + 1 ; l++) {
            for (int i = 0 ; i < this->npl[l -1] + 1 ; i++) {
                for (int j = 1; j < this->npl[l] + 1; j++) {
                    this->weights[l][i][j] -= learning_rate * this->X[l - 1][i] * this->deltas[l][j];
                }
            }
        }


    }
}