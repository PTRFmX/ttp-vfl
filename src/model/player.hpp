#ifndef TTPVFL_PLAYER_HPP
#define TTPVFL_PLAYER_HPP

#include <iostream>
#include "../globals.hpp"
#include "../utils/utils.hpp"
#include "coordinator.hpp"

using namespace std;
using namespace emp;

class Player {

public:

    emp::NetIO *peer_netio, *coord_netio;

    RowMatrixXi64 X;
    ColVectorXi64 w;
    ColVectorXi64 Y;

    RowMatrixXi64 R1, R2;
    ColVectorXi64 wA1, wA2, wB1, wB2;

    ColVectorXi64 alpha1, alpha2, alpha, gamma;

    ColVectorXi64 rand, K, res, t, u, v;

    ColVectorXi64 w_mask, w_temp, t_mask, t_temp;
    ColVectorXi64 l1, l2, l3, l1_mask, l2_mask, k1, k2;
    RowMatrixXi64 X_mask, X_mask1, X_mask2, X_temp;

    void setup();

    void train_linear(double lr);

    void train_linear_DA_simulate(double lr);

    void train_logistic(double lr);

    void train_logistic_DA_simulate(double lr);

    void reshare();

    Player(RowMatrixXi64 X, ColVectorXi64 Y, NetIO* io1, NetIO* io2): 
        X(X), Y(Y), peer_netio(io1), coord_netio(io2) {

        printf("X shape: %d, %d\n", X.rows(), X.cols());
        printf("Y shape: %d, %d\n", Y.rows(), Y.cols());
        
        R1.resize(X.rows(), X.cols());
        R2.resize(X.rows(), X.cols());

        wA1.resize(X.cols(), 1);
        wB1.resize(X.cols(), 1);
        wA2.resize(X.cols(), 1);
        wB2.resize(X.cols(), 1);

        alpha.resize(X.rows(), 1);
        alpha1.resize(X.rows(), 1);
        alpha2.resize(X.rows(), 1);

        rand.resize(X.cols(), 1);
        K.resize(X.rows(), 1);
        res.resize(X.cols(), 1);
        t.resize(X.rows(), 1);

        u.resize(X.rows(), 1);
        v.resize(X.rows(), 1);
        
        w = ColVectorXi64::Random(X.cols(), 1);

        gamma = ColVectorXi64::Random(X.rows(), 1);

        X_mask.resize(X.rows(), X.cols());
        X_mask1 = RowMatrixXi64::Random(X.rows(), X.cols());
        X_mask2 = RowMatrixXi64::Random(X.rows(), X.cols());
        w_mask = ColVectorXi64::Random(X.cols(), 1);
        t_mask = ColVectorXi64::Random(X.rows(), 1);

        X_temp.resize(X.rows(), X.cols());
        w_temp.resize(X.cols(), 1);
        t_temp.resize(X.rows(), 1);

        l1.resize(X.rows(), 1);
        l2.resize(X.rows(), 1);
        l3.resize(X.rows(), 1);

        k1.resize(X.rows(), 1);
        k2.resize(X.rows(), 1);

        l1_mask = ColVectorXi64::Random(X.rows(), 1);
        l2_mask = ColVectorXi64::Random(X.rows(), 1);
        
        setup();
    }
};

#endif