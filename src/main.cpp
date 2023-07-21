#include <stdio.h>
#include <vector>
#include <emp-ot/emp-ot.h>
#include "globals.hpp"
#include "utils/utils.hpp"
#include "tasks/linear_regression.hpp"
#include "tasks/logistic_regression.hpp"

using namespace std;
using namespace emp;

const string s1 = "10.2.1.1", s2 = "10.2.2.1", s3 = "10.2.3.1";

int NUM_IMAGES = BATCH_SIZE;
int PARTY_NUM, N, D;

double total_comm_time = 0;

int main(int argc, char** argv) {
    int port, num_iters = 1;
    string peer_address, coord_address;

    PARTY_NUM = atoi(argv[1]);
    port = atoi(argv[2]);
    N = atoi(argv[3]);
    D = atoi(argv[4]);

    string type(argv[5]);

    bool train_DA = false;

    if (argc > 6) {
        printf("argument count: %d, enabling DA\n", argc);
        train_DA = true;
    }

    NetIO *io1, *io2;

    if (PARTY_NUM == 3) {
        io1 = new NetIO(nullptr, port + 1);
        io2 = new NetIO(nullptr, port + 2);
    } else {
        io1 = new NetIO(PARTY_NUM == 1 ? nullptr : s1.c_str(), port);
        io2 = new NetIO(s3.c_str(), PARTY_NUM == 1 ? port + 1 : port + 2);
    }

    NUM_IMAGES *= num_iters;

    // try{
    //     int x = -1;
    //     if(argc <= 4)
    //         throw x;
    //     address = argv[4];
    // } catch(int x) {
    //     address = "127.0.0.1";
    // }

    // NetIO* io = new NetIO(PARTY == ALICE ? nullptr : address.c_str(), port);

    // TrainingParams params;
    int num_samples, num_features;

    cout << "========" << endl;
    cout << "Training" << endl;
    cout << "========" << endl;

    vector<vector<uint64_t> > training_data;
    vector<uint64_t> training_labels;

    read_MNIST_data<uint64_t>(true, training_data, num_samples, num_features);
    RowMatrixXi64 X(num_samples, num_features);
    vector2d_to_RowMatrixXi64(training_data, X);
    X *= SCALING_FACTOR;
    X /= 255;

    read_MNIST_labels<uint64_t>(true, training_labels);
    ColVectorXi64 Y(num_samples);
    vector_to_ColVectorXi64(training_labels, Y);
    Y *= SCALING_FACTOR;
    Y /= 10;

    if (type == "l") LinearRegression linear_regression(PARTY_NUM < 3, X, Y, io1, io2, train_DA);
    else LogisticRegression logistic_regression(PARTY_NUM < 3, X, Y, io1, io2, train_DA);

    delete io1, io2;

    return 0;
}