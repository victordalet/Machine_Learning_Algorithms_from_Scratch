//
// Created by victor on 21/04/25.
//

#ifndef LINEARREG_H
#define LINEARREG_H
#include <vector>


class Linear {
public:
    explicit Linear();

    void train(const std::vector<double> &x_data, const std::vector<double> &y_data, int epochs, double learning_rate, bool is_classification);

    double predict(double x_data, bool is_classification) const;

    double m;
    double b;

private:
    static double sigmoid(double x);
};


#endif //LINEARREG_H
