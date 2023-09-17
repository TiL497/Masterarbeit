#include "linalg.cpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "kalman.hpp"
#include <vector> 
#include <cmath>
#include <random>

/**
 * @brief initializeKalman 
 *
 * Function to initialize new kalman hypothese 
 *
 * @param &kalman_tracks[inout]: vector of kalman hypotheses  
 * @param x[in]: x coordinate of center of detection  
 * @param y[in]: y coordinate of center of detection  
 * @param w[in]: width of detection  
 * @param h[in]: height of detection  
 * @param id[in]: unique id   
 *  
 */
void initializeKalman(std::vector<KalmanFilter> &kalman_tracks, float x, float y, float w, float h, int id){
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, 255); // define the range
    KalmanFilter kalman; 
    kalman.x[0] = x; 
    kalman.x[1] = y; 
    kalman.x[2] = w; 
    kalman.x[3] = h; 
    kalman.id = id; 
    kalman.r = distr(gen);
    kalman.g = distr(gen);
    kalman.b = distr(gen); 
    kalman_tracks.push_back(kalman); 
}

/**
 * @brief kalmanPredict 
 *
 * Function to predict new state of kalman hypothese 
 *
 * @param *kalman[inout]: kalman hypothese  
 * @param *u[in]: control input - not used   
 *  
 */
void kalmanPredict(KalmanFilter *kalman, float *u){
    //Update time state
    //x_k =Ax_(k-1) + Bu_(k-1)  
    float temp_mult1[DIM] = {0};
    float temp_mult2[DIM] = {0};
    float temp_mult3[DIM*DIM] = {0};
    float temp_mult4[DIM*DIM] = {0};
    float temp_transpose[DIM*DIM] = {0}; 
    matMul(kalman->A, kalman->x, temp_mult1, 8, 8, 8, 1);   
    matMul(kalman->B, u, temp_mult2, 8, 4, 4, 1);
    matAdd(temp_mult1, temp_mult2, kalman->x, 8, 1);
    //Calculate error covariance
    //P= A*P*A' + Q       
    matMul(kalman->A, kalman->P, temp_mult3, 8, 8, 8, 8); 
    matTranspose(kalman->A, temp_transpose, 8, 8); 
    matMul(temp_mult3, temp_transpose, temp_mult4, 8, 8, 8, 8);
    matAdd(temp_mult4, kalman->Q, kalman->P, 8, 8);
}

/**
 * @brief kalmanUpdate 
 *
 * Function to update kalman hypothese 
 *
 * @param *kalman[inout]: kalman hypothese  
 * @param *z[in]: measurement - from matched detection   
 *  
 */
void kalmanUpdate(KalmanFilter *kalman, float *z){
    //S = H*P*H'+R
    float temp_transpose[4*DIM] = {0}; 
    float temp_mult1[4*DIM] = {0}; 
    float temp_mult2[4*DIM] = {0}; 
    float temp_mult3[4*DIM] = {0};
    float temp_mult4[4] = {0}; 
    float temp_mult5[DIM] = {0};
    float temp_mult6[DIM*DIM] = {0};
    float temp_sub[4] = {0};
    float temp_sub1[DIM*DIM] = {0}; 
    float S[4*4] = {0}; 
    float S_inv[4*4] = {0}; 
    float K[4*DIM] = {0}; 
    //S = H*P*H'+R  
    matTranspose(kalman->H, temp_transpose, 4, 8); 
    matMul(kalman->H, kalman->P, temp_mult1, 4, 8, 8, 8); 
    matMul(temp_mult1, temp_transpose, temp_mult2, 4, 8, 8, 4);
    matAdd(temp_mult2, kalman->R, S, 4, 4);
    //Calculate the Kalman Gain
    //K = P * H'* inv(H*P*H'+R)
    matInverse(S, S_inv); 
    matMul(kalman->P, temp_transpose, temp_mult3, 8, 8, 8, 4);
    matMul(temp_mult3, S_inv, K, 8, 4, 4, 4); 
    //Calculate updated state with measurement and Kalman Gain 
    //x = x + K * (z - H * x)
    matMul(kalman->H, kalman->x, temp_mult4, 4, 8, 8, 1);
    matSub(z, temp_mult4, temp_sub, 4, 1);
    matMul(K, temp_sub, temp_mult5, 8, 4, 4, 1);
    matAdd(kalman->x, temp_mult5, kalman->x, 8, 1); 
    //Update error covariance matrix
    //P = (I-K*H)*P
    matMul(K, kalman->H, temp_mult6, 8, 4, 4, 8);
    matSub(kalman->I, temp_mult6, temp_sub1, 8, 8);
    matMul(temp_sub1, kalman->P, kalman->P, 8, 8, 8, 8);
}