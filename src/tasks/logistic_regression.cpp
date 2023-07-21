#include "logistic_regression.hpp"
#include <chrono>

using namespace std::chrono;

extern int PARTY_NUM;
extern double total_comm_time;

LogisticRegression::LogisticRegression(bool is_player, RowMatrixXi64 X, ColVectorXi64 Y, 
    NetIO* io1, NetIO* io2, bool train_DA): is_player(is_player), train_DA(train_DA) {
    if (is_player) {
        RowMatrixXi64 Xi = RowMatrixXi64::Random(N, D);
        ColVectorXi64 Yi = ColVectorXi64::Random(N, 1);

        player = new Player(Xi, Yi, io1, io2);
    } else {
        coord = new Coordinator(io1, io2);
    }
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        train();
        printf("Iteration %d done ...\n", i);
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    double time = duration_cast<microseconds>(t2 - t1).count() / 1000.00;
    printf("elapsed time: %f\n", time);
    printf("total comm time: %f\n", total_comm_time);
}

void LogisticRegression::train() {
    double learning_rate = 0.01;
    if (train_DA) {
        if (is_player) {
            player->train_logistic_DA_simulate(learning_rate);
        } else {
            // coord->train_linear();
        }
    }
    else {
        if (is_player) {
            player->train_logistic(learning_rate);
            // player->reshare();
        } else {
            coord->train_logistic();
        }
    }
}
