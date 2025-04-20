//
// Created by victor on 20/04/25.
//

#ifndef MLP_H
#define MLP_H
#include <vector>



class MLP {

public:
    explicit MLP(const std::vector<int> &npl);
    std::vector<double> predict(const std::vector<double> &input, bool is_classification);
    void train(const std::vector<std::vector<double>> &all_dataset_inputs,
        const std::vector<std::vector<double>> &all_dataset_outputs,
        int interations_count,
        double learning_rate,
        bool is_classification
    );
    std::vector<int> npl;
    std::vector<std::vector<std::vector<double>>> weights;
    int L;
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> deltas;


private:
    void propagate(const std::vector<double> &inputs, bool is_classification);
};



#endif //MLP_H
