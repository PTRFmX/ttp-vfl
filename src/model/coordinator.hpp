#ifndef TTPVFL_COORDINATOR_HPP
#define TTPVFL_COORDINATOR_HPP

#include <iostream>
#include "../globals.hpp"
#include "../utils/utils.hpp"

using namespace std;
using namespace emp;

class Coordinator {

public:

    emp::NetIO *p1, *p2;

    RowMatrixXi64 X;

    RowMatrixXi64 masked_X_A;
    RowMatrixXi64 masked_X_B;

    ColVectorXi64 masked_w_A2;
    ColVectorXi64 masked_w_B1;

    ColVectorXi64 rand1, rand2, K1, K2, u2, v2, S1, S2;

    ColVectorXi64 player1_res, player2_res;

    void setup();

    void train_linear();

    void train_logistic();

    Coordinator(NetIO* io1, NetIO* io2): p1(io1), p2(io2) {
        X.resize(N, D);
        masked_X_A.resize(X.rows(), X.cols());
        masked_X_B.resize(X.rows(), X.cols());
        masked_w_A2.resize(X.cols(), 1);
        masked_w_B1.resize(X.cols(), 1);
        K1 = ColVectorXi64::Random(X.rows(), 1);
        rand1 = ColVectorXi64::Random(X.cols(), 1);
        rand2 = ColVectorXi64::Random(X.cols(), 1);
        u2 = ColVectorXi64::Random(X.rows(), 1);
        v2 = ColVectorXi64::Random(X.rows(), 1);
        S1.resize(X.rows(), 1);
        S2.resize(X.rows(), 1);
        player1_res.resize(X.rows(), 1);
        player2_res.resize(X.rows(), 1);

        setup();
    }

};

#endif