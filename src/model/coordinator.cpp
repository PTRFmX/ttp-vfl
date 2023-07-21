#include "coordinator.hpp"
#include <Eigen/Dense>
#include <thread>

#include <chrono>

using namespace Eigen;
using namespace std::chrono;

extern int PARTY_NUM;

void Coordinator::setup() {
    recv<RowMatrixXi64>(p1, masked_X_A);
    recv<ColVectorXi64>(p1, masked_w_A2);
    p1->flush();
    recv<RowMatrixXi64>(p2, masked_X_B);
    recv<ColVectorXi64>(p2, masked_w_B1);
    p2->flush();
    // p1->recv_data_internal((void*) &masked_X_A, sizeof(masked_X_A));
    // p1->recv_data_internal((void*) &masked_w_A2, sizeof(masked_w_A2));
    // p2->recv_data_internal((void*) &masked_X_B, sizeof(masked_X_B));
    // p2->recv_data_internal((void*) &masked_w_B1, sizeof(masked_w_B1));
}

void Coordinator::train_linear() {

    std::thread A1(recv<ColVectorXi64>, p1, std::ref(player1_res));
    std::thread A2(recv<ColVectorXi64>, p2, std::ref(player2_res));

    // high_resolution_clock::time_point t1 = high_resolution_clock::now();

    ColVectorXi64 masked_loss = masked_X_A * masked_w_A2 + masked_X_B * masked_w_B1;

    // high_resolution_clock::time_point t2 = high_resolution_clock::now();

    A1.join();
    A2.join();

    // Distribute

    // high_resolution_clock::time_point t3 = high_resolution_clock::now();

    K2 = masked_loss + player1_res + player2_res - K1;

    // high_resolution_clock::time_point t4 = high_resolution_clock::now();

    // thread A1(send<ColVectorXi64>, p1, K1);
    // thread A2(send<ColVectorXi64>, p1, rand1);

    send<ColVectorXi64>(p1, K1);
    send<ColVectorXi64>(p1, rand1);
    
    send<ColVectorXi64>(p2, K2);
    send<ColVectorXi64>(p2, rand2);

    // high_resolution_clock::time_point t5 = high_resolution_clock::now();

    ColVectorXi64 C1 = masked_X_B.transpose() * K1 + rand2, C2 = masked_X_A.transpose() * K2 + rand1;

    truncate<ColVectorXi64>(PARTY_NUM, SCALING_FACTOR, C1);
    truncate<ColVectorXi64>(PARTY_NUM, SCALING_FACTOR, C2);

    // high_resolution_clock::time_point t6 = high_resolution_clock::now();

    // thread A3(send<ColVectorXi64>, p1, C1);
    // thread B3(send<ColVectorXi64>, p2, C2);
    send<ColVectorXi64>(p1, C1);
    send<ColVectorXi64>(p2, C2);
}

void Coordinator::train_logistic() {

    std::thread A1(recv<ColVectorXi64>, p1, std::ref(player1_res));
    std::thread A2(recv<ColVectorXi64>, p2, std::ref(player2_res));

    // high_resolution_clock::time_point t1 = high_resolution_clock::now();

    ColVectorXi64 masked_loss = masked_X_A * masked_w_A2 + masked_X_B * masked_w_B1;

    // high_resolution_clock::time_point t2 = high_resolution_clock::now();

    A1.join();
    A2.join();

    // Distribute

    // high_resolution_clock::time_point t3 = high_resolution_clock::now();

    ColVectorXi64 Z = masked_loss + player1_res + player2_res;

    ColVectorXi64 Z3 = Z.array().square() * Z.array();
    ColVectorXi64 u1 = Z3 - u2;
    ColVectorXi64 v1 = Z - v2;

    send<ColVectorXi64>(p1, u1);
    send<ColVectorXi64>(p2, u2);
    send<ColVectorXi64>(p1, v1);
    send<ColVectorXi64>(p2, v2);

    recv<ColVectorXi64>(p1, S1);
    recv<ColVectorXi64>(p2, S2);

    K2 = S1 + S2 - K1;

    // high_resolution_clock::time_point t4 = high_resolution_clock::now();

    // thread A1(send<ColVectorXi64>, p1, K1);
    // thread A2(send<ColVectorXi64>, p1, rand1);

    send<ColVectorXi64>(p1, K1);
    send<ColVectorXi64>(p1, rand1);
    
    send<ColVectorXi64>(p2, K2);
    send<ColVectorXi64>(p2, rand2);

    // high_resolution_clock::time_point t5 = high_resolution_clock::now();

    ColVectorXi64 C1 = masked_X_B.transpose() * K1 + rand2, C2 = masked_X_A.transpose() * K2 + rand1;

    // high_resolution_clock::time_point t6 = high_resolution_clock::now();

    // thread A3(send<ColVectorXi64>, p1, C1);
    // thread B3(send<ColVectorXi64>, p2, C2);
    send<ColVectorXi64>(p1, C1);
    send<ColVectorXi64>(p2, C2);
}