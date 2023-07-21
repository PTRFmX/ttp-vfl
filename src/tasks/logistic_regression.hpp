#ifndef TTPVFL_LOGR_HPP
#define TTPVFL_LOGR_HPP
#include "../model/player.hpp"
#include "../globals.hpp"

class LogisticRegression {
public:

    Coordinator* coord;
    Player* player;
    bool is_player, train_DA;

    LogisticRegression(bool is_player, RowMatrixXi64 X, ColVectorXi64 Y, NetIO* io1, NetIO* io2, bool train_DA);

    ~LogisticRegression() {
        if (is_player) delete player;
        else delete coord;
    }

    void train();
    // void test(RowMatrixXd& testing_data, ColVectorXd& testing_labels);
};
#endif
