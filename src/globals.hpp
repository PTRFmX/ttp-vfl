#ifndef TTPVFL_GLOBAL_HPP
#define TTPVFL_GLOBAL_HPP
#include <Eigen/Dense>
#include <emp-tool/emp-tool.h>

#define BATCH_SIZE 128
#define BITLEN 64
#define LEARNING_RATE_INV 128 // 1/LEARNING_RATE
#define DEBUG 1

// #define N 1024
// #define D 3000

#define SCALING_FACTOR 8192 // Precision of 13 bits

extern int PARTY;
extern int N;
extern int D;

using namespace Eigen;

typedef Matrix<uint64_t, Dynamic, Dynamic, RowMajor> RowMatrixXi64;
typedef Matrix<uint64_t, Dynamic, Dynamic, ColMajor> ColMatrixXi64;
typedef Matrix<uint64_t, 1, Dynamic, RowMajor> RowVectorXi64;
typedef Matrix<uint64_t, Dynamic, 1, ColMajor> ColVectorXi64;
typedef Matrix<int64_t, Dynamic, Dynamic, RowMajor> SignedRowMatrixXi64;
typedef Matrix<int64_t, Dynamic, Dynamic, ColMajor> SignedColMatrixXi64;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrixXd;
typedef Matrix<double, Dynamic, Dynamic, ColMajor> ColMatrixXd;
typedef Matrix<double, 1, Dynamic, RowMajor> RowVectorXd;
typedef Matrix<double, Dynamic, 1, ColMajor> ColVectorXd;

#endif
