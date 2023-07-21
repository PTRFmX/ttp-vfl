#include "player.hpp"
#include <chrono>

using namespace std::chrono;

extern int PARTY_NUM;
extern double total_comm_time;

void Player::setup() {
    if (PARTY_NUM == 1) {

        R1 = RowMatrixXi64::Random(X.rows(), X.cols());
        wA2 = ColVectorXi64::Random(X.cols(), 1);

        send<RowMatrixXi64>(peer_netio, R1);
        send<ColVectorXi64>(peer_netio, wA2);
        // peer_netio->send_data_internal((void*)&R1, sizeof(R1));
        // peer_netio->send_data_internal((void*)&wA2, sizeof(wA2));

        peer_netio->flush();

        // recv_data_internal from peer R2, wB1
        recv<RowMatrixXi64>(peer_netio, R2);
        recv<ColVectorXi64>(peer_netio, wB1);
        // peer_netio->recv_data_internal((void*)&R2, sizeof(R2));
        // peer_netio->recv_data_internal((void*)&wB1, sizeof(wB1));
        
        peer_netio->flush();

        // send_data_internal to coordinator X - R1, wA2
        RowMatrixXi64 t = X - R1;
        send<RowMatrixXi64>(coord_netio, t);
        send<ColVectorXi64>(coord_netio, wA2);

        coord_netio->flush();
        // coord_netio->send_data_internal((void*)&t, sizeof(t));
        // coord_netio->send_data_internal((void*)&wA2, sizeof(wA2));

    } else if (PARTY_NUM == 2) {

        R2 = RowMatrixXi64::Random(X.rows(), X.cols());
        wB1 = ColVectorXi64::Random(X.cols(), 1);

        recv<RowMatrixXi64>(peer_netio, R1);
        recv<ColVectorXi64>(peer_netio, wA2);
        // peer_netio->recv_data_internal((void*)&R1, sizeof(R1));
        // peer_netio->recv_data_internal((void*)&wA2, sizeof(wA2));

        peer_netio->flush();

        send<RowMatrixXi64>(peer_netio, R2);
        send<ColVectorXi64>(peer_netio, wB1);
        // peer_netio->send_data_internal((void*)&R2, sizeof(R2));
        // peer_netio->send_data_internal((void*)&wB1, sizeof(wB1));
        
        peer_netio->flush();

        // send_data_internal to coordinator X - R1, wA2
        RowMatrixXi64 t = X - R2;
        send<RowMatrixXi64>(coord_netio, t);
        send<ColVectorXi64>(coord_netio, wB1);

        coord_netio->flush();
        // coord_netio->send_data_internal((void*)&t, sizeof(t));
        // coord_netio->send_data_internal((void*)&wB1, sizeof(wB1));
    }
}

void Player::train_linear(double lr) {
    // high_resolution_clock::time_point t1 = high_resolution_clock::now();
    if (PARTY_NUM == 1) {
        // Compute local result and send_data_internal to coordinator
        t = X * wA1 + R2 * wB1 + alpha1;
    } else if (PARTY_NUM == 2) {
        // Compute local result and send_data_internal to coordinator
        t = X * wB2 + R1 * wA2 - Y + alpha2;
    }

    truncate<ColVectorXi64>(PARTY_NUM, SCALING_FACTOR, t);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    send<ColVectorXi64>(coord_netio, t);
    recv<ColVectorXi64>(coord_netio, K);
    recv<ColVectorXi64>(coord_netio, rand);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    high_resolution_clock::time_point t3, t4;
    if (PARTY_NUM == 1) {
        wA1 += X.transpose() * (K - alpha) - rand;
        wB1 += R2.transpose() * K;
        t3 = high_resolution_clock::now();
        recv<ColVectorXi64>(coord_netio, res);
        t4 = high_resolution_clock::now();
        wA1 += res;
        truncate<ColVectorXi64>(PARTY_NUM, SCALING_FACTOR, wA1);
        truncate<ColVectorXi64>(PARTY_NUM, SCALING_FACTOR, wB1);

    } else {
        wB2 += X.transpose() * (K - alpha) - rand;
        wA2 += R1.transpose() * K;
        t3 = high_resolution_clock::now();
        recv<ColVectorXi64>(coord_netio, res);
        t4 = high_resolution_clock::now();
        wB2 += res;
        truncate<ColVectorXi64>(PARTY_NUM, SCALING_FACTOR, wA2);
        truncate<ColVectorXi64>(PARTY_NUM, SCALING_FACTOR, wB2);
    }
    // high_resolution_clock::time_point t4 = high_resolution_clock::now();

    double comm_time = duration_cast<microseconds>((t2 - t1) + (t4 - t3)).count() / 1000.00;
    total_comm_time += comm_time;
    // printf("communication time: %f ms\n", comm_time);
}

void Player::reshare() {
    ColVectorXi64 rand_mask;
    ColVectorXi64 wA, wB;
    if (PARTY_NUM == 1) {
        wA = wA1 + rand_mask;
        wB = wB1 + rand_mask;
    } else if (PARTY_NUM == 2) {
        wA = wA2 - rand_mask;
        wB = wB2 - rand_mask;
    }
    send<ColVectorXi64>(coord_netio, wA);
    send<ColVectorXi64>(coord_netio, wB);
}

void Player::train_logistic(double lr) {
    high_resolution_clock::time_point t0 = high_resolution_clock::now();

    if (PARTY_NUM == 1) {
        // Compute local result and send_data_internal to coordinator
        t = X * wA1 + R2 * wB1;
    } else if (PARTY_NUM == 2) {
        // Compute local result and send_data_internal to coordinator
        t = X * wB2 + R1 * wA2 + alpha2;
    }

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    send<ColVectorXi64>(coord_netio, t);

    recv<ColVectorXi64>(coord_netio, u);
    recv<ColVectorXi64>(coord_netio, v);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    ColVectorXi64 S;

    if (PARTY_NUM == 1) {
        S = gamma.array() * u.array() + gamma.array() * v.array();
        S += alpha1;
    } else if (PARTY_NUM == 2) {
        S = gamma.array() * u.array() + gamma.array() * v.array();
        S += alpha2 - Y;
    }

    high_resolution_clock::time_point t3 = high_resolution_clock::now();

    send<ColVectorXi64>(coord_netio, S);

    recv<ColVectorXi64>(coord_netio, K);
    recv<ColVectorXi64>(coord_netio, rand);

    high_resolution_clock::time_point t4 = high_resolution_clock::now();

    high_resolution_clock::time_point t5, t6;

    if (PARTY_NUM == 1) {
        wA1 += X.transpose() * (K - alpha) - rand;
        wB1 += R2.transpose() * K;
        t5 = high_resolution_clock::now();
        recv<ColVectorXi64>(coord_netio, res);
        t6 = high_resolution_clock::now();
        wA1 += res;
    } else {
        wB2 += X.transpose() * (K - alpha) - rand;
        wA2 += R1.transpose() * K;
        t5 = high_resolution_clock::now();
        recv<ColVectorXi64>(coord_netio, res);
        t6 = high_resolution_clock::now();
        wB2 += res;
    }
    // high_resolution_clock::time_point t4 = high_resolution_clock::now();

    // double time1 = duration_cast<microseconds>((t2 - t1)).count() / 1000.00;
    // double time2 = duration_cast<microseconds>((t3 - t2)).count() / 1000.00;
    // double time3 = duration_cast<microseconds>((t4 - t3)).count() / 1000.00;
    // printf("time1: %f\n", time1);
    // printf("time2: %f\n", time2);
    double comm_time = duration_cast<microseconds>((t2 - t1) + (t4 - t3) + (t6 - t5)).count() / 1000.00;

    total_comm_time += comm_time;
    // printf("communication time: %f ms\n", comm_time);
}

void Player::train_linear_DA_simulate(double lr) {

    high_resolution_clock::time_point t0, t1, t2, t3;
    double forward_comm, backward_comm;

    double total_comp;
    t0 = high_resolution_clock::now();
    if (PARTY_NUM == 1) {

        X_mask = X + X_mask1;
        w_mask = w + w_mask;

        t1 = high_resolution_clock::now();

        send<RowMatrixXi64>(peer_netio, X_mask);
        recv<ColVectorXi64>(peer_netio, w_temp);

        recv<RowMatrixXi64>(peer_netio, X_temp);
        send<ColVectorXi64>(peer_netio, w_mask);

        peer_netio->flush();
        
        t2 = high_resolution_clock::now();

        // double time0 = duration_cast<microseconds>((t3 - t2)).count() / 1000.00;
        // printf("forward comm recv time: %f\n", time0);
        
        // printf("forward comm: %f\n", time1);

        t = X * wA1 + X_mask * w_temp + X_temp * w_mask;

    } else if (PARTY_NUM == 2) {

        X_mask = X + X_mask1;
        w_mask = w + w_mask;

        t1 = high_resolution_clock::now();
        
        recv<RowMatrixXi64>(peer_netio, X_temp);
        send<ColVectorXi64>(peer_netio, w_mask);
        
        send<RowMatrixXi64>(peer_netio, X_mask);
        recv<ColVectorXi64>(peer_netio, w_temp);

        peer_netio->flush();

        t2 = high_resolution_clock::now();

        t = X * wB2 + X_mask * w_temp + X_temp * w_mask - Y;
    }
    t3 = high_resolution_clock::now();

    forward_comm = duration_cast<microseconds>((t2 - t1)).count() / 1000.00;

    total_comp = duration_cast<microseconds>((t3 - t2) + (t1 - t0)).count() / 1000.00;
    printf("forward computation time: %f ms\n", total_comp);

    t0 = high_resolution_clock::now();

    if (PARTY_NUM == 1) {

        X_mask = X + X_mask2;
        t_mask = t + t_mask;

        t1 = high_resolution_clock::now();

        send<RowMatrixXi64>(peer_netio, X_mask);
        recv<ColVectorXi64>(peer_netio, t_temp);

        recv<RowMatrixXi64>(peer_netio, X_temp);
        send<ColVectorXi64>(peer_netio, t_mask);

        peer_netio->flush();

        t2 = high_resolution_clock::now();
        
        wA1 += X.transpose() * t + X_mask.transpose() * t_temp;
        wB1 += X_temp.transpose() * t_mask;

    } else if (PARTY_NUM == 2) {

        X_mask = X + X_mask2;
        t_mask = t + t_mask;

        t1 = high_resolution_clock::now();

        recv<RowMatrixXi64>(peer_netio, X_temp);
        send<ColVectorXi64>(peer_netio, t_mask);
        
        send<RowMatrixXi64>(peer_netio, X_mask);
        recv<ColVectorXi64>(peer_netio, t_temp);

        peer_netio->flush();

        t2 = high_resolution_clock::now();

        wB2 += X.transpose() * t + X_mask.transpose() * t_temp;
        wA2 += X_temp.transpose() * t_mask;
    }
    
    t3 = high_resolution_clock::now();

    backward_comm = duration_cast<microseconds>((t2 - t1)).count() / 1000.00;
    total_comm_time += forward_comm + backward_comm;

    total_comp += duration_cast<microseconds>((t3 - t2) + (t1 - t0)).count() / 1000.00;
    printf("total computation time: %f ms\n", total_comp);
    // printf("communication time: %f ms\n", forward_comm + backward_comm);
}

void Player::train_logistic_DA_simulate(double lr) {

    high_resolution_clock::time_point t1, t2;
    double forward_comm, backward_comm;
    
    if (PARTY_NUM == 1) {

        X_mask = X + X_mask1;
        w_mask = w + w_mask;

        t1 = high_resolution_clock::now();

        send<RowMatrixXi64>(peer_netio, X_mask);
        recv<ColVectorXi64>(peer_netio, w_temp);

        recv<RowMatrixXi64>(peer_netio, X_temp);
        send<ColVectorXi64>(peer_netio, w_mask);

        t2 = high_resolution_clock::now();

        l1 = X * wA1 + X_mask * w_temp + X_temp * w_mask;
    } else if (PARTY_NUM == 2) {

        X_mask = X + X_mask1;
        w_mask = w + w_mask;

        t1 = high_resolution_clock::now();
        
        recv<RowMatrixXi64>(peer_netio, X_temp);
        send<ColVectorXi64>(peer_netio, w_mask);
        
        send<RowMatrixXi64>(peer_netio, X_mask);
        recv<ColVectorXi64>(peer_netio, w_temp);

        t2 = high_resolution_clock::now();

        l1 = X * wB2 + X_mask * w_temp + X_temp * w_mask;
    }

    forward_comm = duration_cast<microseconds>((t2 - t1)).count() / 1000.00;

    // sigmoid(l + k) ~= c0(l + k)^3 + c1(l + k) + c2

    l2 = l1.array().square();
    l3 = l2.array() * l1.array();

    l2_mask = l2 + l2_mask;
    l1_mask = l1 + l1_mask;

    t1 = high_resolution_clock::now();
    
    if (PARTY_NUM == 1) {
        
        send<ColVectorXi64>(peer_netio, l2_mask);
        recv<ColVectorXi64>(peer_netio, k1);
        
        peer_netio->flush();
        
        recv<ColVectorXi64>(peer_netio, k2);
        send<ColVectorXi64>(peer_netio, l1_mask);
        
        peer_netio->flush();     

    } else {
        
        recv<ColVectorXi64>(peer_netio, k2);
        send<ColVectorXi64>(peer_netio, l1_mask);
        
        peer_netio->flush();
        
        send<ColVectorXi64>(peer_netio, l2_mask);
        recv<ColVectorXi64>(peer_netio, k1);
        
        peer_netio->flush();
    }

    t2 = high_resolution_clock::now();

    forward_comm += duration_cast<microseconds>((t2 - t1)).count() / 1000.00;

    t = l2_mask.array() * k1.array() + l1_mask.array() * k2.array();
    t += l3;

    // DA partial cleartext backward gradient calculation
    if (PARTY_NUM == 1) {

        X_mask = X + X_mask2;
        t_mask = t + t_mask;

        t1 = high_resolution_clock::now();

        send<RowMatrixXi64>(peer_netio, X_mask);
        recv<ColVectorXi64>(peer_netio, t_temp);

        peer_netio->flush();

        recv<RowMatrixXi64>(peer_netio, X_temp);
        send<ColVectorXi64>(peer_netio, t_mask);

        peer_netio->flush();

        t2 = high_resolution_clock::now();

        wA1 += X.transpose() * t + X_mask.transpose() * t_temp;
        wB1 += X_temp.transpose() * t_mask;

    } else if (PARTY_NUM == 2) {

        X_mask = X + X_mask2;
        t_mask = t + t_mask;

        t1 = high_resolution_clock::now();

        recv<RowMatrixXi64>(peer_netio, X_temp);
        send<ColVectorXi64>(peer_netio, t_mask);

        peer_netio->flush();
        
        send<RowMatrixXi64>(peer_netio, X_mask);
        recv<ColVectorXi64>(peer_netio, t_temp);

        peer_netio->flush();

        t2 = high_resolution_clock::now();

        wB2 += X.transpose() * t + X_mask.transpose() * t_temp;
        wA2 += X_temp.transpose() * t_mask;
    }

    backward_comm = duration_cast<microseconds>((t2 - t1)).count() / 1000.00;
    total_comm_time += forward_comm + backward_comm;
    // printf("communication time: %f ms\n", forward_comm + backward_comm);
}