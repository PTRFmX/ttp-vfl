#ifndef TTPVFL_UTILS_HPP
#define TTPVFL_UTILS_HPP
#include "../globals.hpp"
#include <iostream>
#include <vector>

extern int NUM_IMAGES;

void vector2d_to_RowMatrixXd(std::vector<std::vector<double>>& x, RowMatrixXd& X);
void vector_to_ColVectorXd(std::vector<double>& x, ColVectorXd& X);
void vector_to_RowVectorXi64(std::vector<uint64_t>& x, RowVectorXi64& X);
void vector2d_to_RowMatrixXi64(std::vector<std::vector<uint64_t>>& x, RowMatrixXi64& X);
void vector2d_to_ColMatrixXi64(std::vector<std::vector<uint64_t>>& x, ColMatrixXi64& X);
void vector_to_ColVectorXi64(std::vector<uint64_t>& x, ColVectorXi64& X);
void RowMatrixXi64_to_vector2d(RowMatrixXi64 X, std::vector<std::vector<uint64_t>>& x);
std::vector<uint64_t> ColVectorXi64_to_vector(ColVectorXi64 X);

void print128_num(emp::block var);
void print_binary(uint64_t int_);
void int_to_bool(bool* bool_, uint64_t int_);

uint64_t extract_lo64(__m128i x);
uint64_t extract_hi64(__m128i x);

int reverse_int(int i);

template<class Derived>
void send(emp::NetIO* io, Eigen::PlainObjectBase<Derived>& X){
    io->send_data(X.data(), X.rows() * X.cols() * sizeof(uint64_t));
    return;
}

template<class Derived>
void recv(emp::NetIO* io, Eigen::PlainObjectBase<Derived>& X){
    io->recv_data(X.data(), X.rows() * X.cols() * sizeof(uint64_t));
    return;
}

template<class Derived, class OtherDerived>
void scale(Eigen::PlainObjectBase<Derived>& X, Eigen::PlainObjectBase<OtherDerived>& x){
    Derived scaled_X = X * SCALING_FACTOR;
    x = scaled_X.template cast<uint64_t>();
    return;
}

template<class Derived, class OtherDerived>
void descale(Eigen::PlainObjectBase<Derived>& X, Eigen::PlainObjectBase<OtherDerived>& x){
    Derived signed_X = X * SCALING_FACTOR;
    x = (X.template cast<int64_t>()).template cast<double>();
    x /= SCALING_FACTOR;
    return;
}

template<class Derived>
void truncate(int i, uint64_t scaling_factor, Eigen::PlainObjectBase<Derived>& X){
    // if (i == 1)
    //     X = -1 * X;
    X /= scaling_factor;
    // if (i == 1)
        // X = -1 * X;
    return;
}

template <typename T>
void read_MNIST_data(bool train, std::vector<std::vector<T>> &vec, int& number_of_images, int& number_of_features){
    std::ifstream file;
    if (train == true)
        file.open("../data/train-images-idx3-ubyte",std::ios::binary);
    else
        file.open("../data/t10k-images-idx3-ubyte",std::ios::binary);
    
    if(!file){
        printf("open file failed! check data path\n");
        exit(1);
    }
    if (file.is_open()){
        int magic_number = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        file.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        if(train == true)
            number_of_images = NUM_IMAGES;
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = reverse_int(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = reverse_int(n_cols);
        number_of_features = n_rows * n_cols;
        std::cout << "Number of Images: " << number_of_images << std::endl;
        std::cout << "Number of Features: " << number_of_features << std::endl;
        for(int i = 0; i < number_of_images; ++i){
            std::vector<T> tp;
            for(int r = 0; r < n_rows; ++r)
                for(int c = 0; c < n_cols; ++c){
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.push_back(((T) temp));
                }
            vec.push_back(tp);
        }
    }
}


template <typename T>
void read_MNIST_labels(bool train, std::vector<T> &vec){
    std::ifstream file;
    if (train == true)
        file.open("../data/train-labels-idx1-ubyte",std::ios::binary);
    else
        file.open("../data/t10k-labels-idx1-ubyte", std::ios::binary);
    if(!file){
        printf("open file failed! check data path\n");
        exit(1);
    }
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        file.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        if(train == true)
            number_of_images = NUM_IMAGES;
        for(int i = 0; i < number_of_images; ++i){
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            if((T) temp == 0)
                vec.push_back((T) 0);
            else
                vec.push_back((T) 1);
        }
    }
}

#endif
